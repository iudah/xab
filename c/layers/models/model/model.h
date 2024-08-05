#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "../../../core/base.h"

XABAPI void xab_model_add(void *model_instance, const void *layer);
XABAPI void xab_configure(void *layer, ...) __attribute__((sentinel));
XABAPI void *xab_model_predict(void *model, void *x, ...)
    __attribute__((sentinel));
XABAPI void xab_model_fit(void *model, void *x, void *y, uint64_t epochs,
                          uint64_t batch);

#endif
