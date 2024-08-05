#ifndef ADAM_R_H
#define ADAM_R_H

#include <stdint.h>

#include <TypeClass.r.h>
#include <Ubject.r.h>
#include <laud.h>

#include "../optimizer/optimizer.r.h"

struct xab_adam {

#ifndef XAB_ADAM_PRIVATE
  char ___[sizeof(struct xab_adam_private {
#endif
    struct xab_optimizer _;

    void **m;
    void **s;
    uint64_t m_length;
    number_t learning_rate;
    number_t beta1;
    number_t beta2;
    number_t eps;
    number_t decay;

#ifndef XAB_ADAM_PRIVATE
  })];
#endif
};

struct xab_adam_class {

#ifndef XAB_ADAM_PRIVATE
  char ___[sizeof(struct XAB_ADAM_CLASS_ {
#endif
    struct xab_optimizer_class _;

#ifndef XAB_ADAM_PRIVATE
  })];
#endif
};

#define XAB_ADAM_PRIORITY (XAB_OPTIMIZER_PRIORITY + 3)

extern const void *XABAdam;
extern const void *XABAdamClass;

#ifdef XAB_ADAM_PROTECTED
#endif

#endif