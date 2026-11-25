#ifndef SOFTMAX_MMIO_H
#define SOFTMAX_MMIO_H

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define SOFTMAX_LENGTH_OFFSET 0x100u
#define SOFTMAX_SCORE_OFFSET  0x110u
#define SOFTMAX_RESULT_OFFSET 0x120u
#define SOFTMAX_OBUF_OFFSET   0x130u
#define SOFTMAX_PACKED_OFFSET 0x200u
#define SOFTMAX_PACKED_MODE   (1u << 24)
#define SOFTMAX_PACKED_DONE   (1u << 13)
#define SOFTMAX_VALID         (1u << 8)
#define SOFTMAX_LAST          (1u << 9)
#define SOFTMAX_OBUF_ERROR    (1u << 12)
#define SOFTMAX_MAX_LENGTH    196u
#define SOFTMAX_SCORE_SCALE   0.11335523860661063
#define SOFTMAX_PROB_SCALE    (1.0 / 127.0)

/* All accesses use the same device mapping as the accelerator registers. */
static inline void softmax_write(volatile uint8_t *accel, size_t offset, uint32_t value)
{
    *(volatile uint32_t *)(accel + offset) = value;
    __sync_synchronize();
}

/* poll_limit is the maximum number of MMIO reads per probability. */
static inline int softmax_read_row(volatile uint8_t *accel, int8_t *probabilities,
                                   size_t length, unsigned poll_limit)
{
    volatile uint32_t *result = (volatile uint32_t *)(accel + SOFTMAX_RESULT_OFFSET);
    for (size_t key = 0; key < length; ++key) {
        uint32_t word = 0;
        for (unsigned poll = 0; poll < poll_limit; ++poll) {
            word = *result;
            if (word & (SOFTMAX_VALID | SOFTMAX_OBUF_ERROR))
                break;
        }
        if ((word & SOFTMAX_OBUF_ERROR) || !(word & SOFTMAX_VALID) ||
            !!(word & SOFTMAX_LAST) != (key == length - 1)) {
            softmax_write(accel, SOFTMAX_LENGTH_OFFSET, 0);
            return (word & (SOFTMAX_VALID | SOFTMAX_OBUF_ERROR)) ? -EIO : -ETIMEDOUT;
        }
        probabilities[key] = (int8_t)(word & 0x7fu);
    }
    return 0;
}

/* One caller owns the device for the duration of a row. */
static inline int softmax_row_mmio(volatile uint8_t *accel, const int8_t *scores,
                                   int8_t *probabilities, size_t length,
                                   unsigned poll_limit)
{
    if (length == 0 || length > SOFTMAX_MAX_LENGTH || poll_limit == 0)
        return -EINVAL;

    softmax_write(accel, SOFTMAX_LENGTH_OFFSET, 0);
    softmax_write(accel, SOFTMAX_LENGTH_OFFSET, (uint32_t)length);
    for (size_t key = 0; key < length; ++key)
        softmax_write(accel, SOFTMAX_SCORE_OFFSET, (uint8_t)scores[key]);
    return softmax_read_row(accel, probabilities, length, poll_limit);
}

/* canvas_offset is the byte offset of a padded row within OBUF. After the
 * producer finishes, its scores must already be in the softmax input units. */
static inline int softmax_obuf_row_mmio(volatile uint8_t *accel, size_t canvas_offset,
                                        int8_t *probabilities, size_t length,
                                        unsigned poll_limit)
{
    if (length == 0 || length > SOFTMAX_MAX_LENGTH || poll_limit == 0 ||
        (canvas_offset & 15u) || canvas_offset > 65536u - length)
        return -EINVAL;

    softmax_write(accel, SOFTMAX_OBUF_OFFSET, ((uint32_t)length << 16) |
                                           (uint32_t)canvas_offset);
    return softmax_read_row(accel, probabilities, length, poll_limit);
}

/* After completion the packed aperture accepts aligned Avalon bursts. */
static inline int softmax_obuf_wait_packed(volatile uint8_t *accel, size_t canvas_offset,
                                           size_t length, unsigned poll_limit)
{
    if (length == 0 || length > SOFTMAX_MAX_LENGTH || poll_limit == 0 ||
        (canvas_offset & 15u) || canvas_offset > 65536u - length)
        return -EINVAL;

    softmax_write(accel, SOFTMAX_OBUF_OFFSET, SOFTMAX_PACKED_MODE |
                                           ((uint32_t)length << 16) |
                                           (uint32_t)canvas_offset);
    volatile uint32_t *status = (volatile uint32_t *)(accel + SOFTMAX_RESULT_OFFSET);
    uint32_t word = 0;
    for (unsigned poll = 0; poll < poll_limit; ++poll) {
        word = *status;
        if (word & (SOFTMAX_PACKED_DONE | SOFTMAX_OBUF_ERROR))
            break;
    }
    if ((word & SOFTMAX_OBUF_ERROR) || !(word & SOFTMAX_PACKED_DONE)) {
        softmax_write(accel, SOFTMAX_LENGTH_OFFSET, 0);
        return (word & SOFTMAX_OBUF_ERROR) ? -EIO : -ETIMEDOUT;
    }
    return 0;
}

static inline int softmax_obuf_packed_mmio(volatile uint8_t *accel, size_t canvas_offset,
                                          int8_t *probabilities, size_t length,
                                          unsigned poll_limit)
{
    int status = softmax_obuf_wait_packed(accel, canvas_offset, length, poll_limit);
    if (status)
        return status;
    volatile uint32_t *packed = (volatile uint32_t *)(accel + SOFTMAX_PACKED_OFFSET);
    for (size_t key = 0; key < length;) {
        uint32_t word = packed[key / 4];
        for (unsigned byte = 0; byte < 4 && key < length; ++byte, ++key) {
            probabilities[key] = (int8_t)(word & 0xffu);
            word >>= 8;
        }
    }
    return 0;
}

/* Convert one TinyViT query row. qk_scale is the real value of one OBUF
 * code before key-dimension scaling. bias[key] is the learned real-valued
 * attention bias for this head/query/key. */
static inline int tinyvit_softmax_row_mmio(volatile uint8_t *accel,
                                           const int8_t *qk, double qk_scale,
                                           double attention_scale, const float *bias,
                                           int8_t *probabilities, size_t length,
                                           unsigned poll_limit)
{
    int8_t scores[SOFTMAX_MAX_LENGTH];
    if (length == 0 || length > SOFTMAX_MAX_LENGTH)
        return -EINVAL;

    for (size_t key = 0; key < length; ++key) {
        double scaled = (qk[key] * qk_scale * attention_scale + bias[key])
                        / SOFTMAX_SCORE_SCALE;
        if (!isfinite(scaled))
            return -EINVAL;
        scaled = fmin(127.0, fmax(-128.0, scaled));
        double lower = floor(scaled);
        double fraction = scaled - lower;
        if (fraction > 0.5 || (fraction == 0.5 && (int)lower % 2 != 0))
            lower += 1.0;
        scores[key] = (int8_t)lower;
    }
    return softmax_row_mmio(accel, scores, probabilities, length, poll_limit);
}

#endif
