#include <Ubject.h>
#include <Ubject.r.h>
#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include <laud_nn.h>
#include <xab_defines.h>

#define XAB_OPTIMIZER_PRIVATE
#define XAB_OPTIMIZER_PROTECTED
#include "../../layers/base_layer/layer.r.h"
#include "optimizer.r.h"

const void *XABOptimizerClass = 0;
const void *XABOptimizer = 0;

static void *optimizer_class_ctor(void *optimizer_class, va_list *arg);

static void *optimizer_ctor(void *optimizer, va_list *arg);

static void *optimizer_dtor(void *self);

static void optimize(void *self, void **weights);

static void fini_optimizer();

static void __attribute__((constructor(XAB_OPTIMIZER_PRIORITY)))
init_optimizer(void) {
  if (!XABOptimizerClass) {
    XABOptimizerClass =
        init(TypeClass, TypeClass, sizeof(struct xab_optimizer_class), //
             ctor, optimizer_class_ctor,                               //
             NULL);
  }

  if (!XABOptimizer) {
    XABOptimizer =
        init(XABOptimizerClass, Ubject, sizeof(struct xab_optimizer), //
             className, "XABOptimizer",                               //
             ctor, optimizer_ctor,                                    //
             dtor, optimizer_dtor,                                    //
             xab_optimize, optimize,                                  //
             NULL);
  }

  atexit(fini_optimizer);
}

static void fini_optimizer() {
  FREE((void *)XABOptimizer);
  FREE((void *)XABOptimizerClass);
}

static void *optimizer_class_ctor(void *self, va_list *arg) {
  struct xab_optimizer_class *optimizer_class =
      super_ctor(XABOptimizerClass, self, arg);

  typedef void (*voidf)();

  voidf selector;
  va_list args = *arg;

  while ((selector = va_arg(args, voidf))) {
    voidf method = va_arg(args, voidf);

    if (selector == (voidf)xab_optimize)
      memcpy(&optimizer_class->optimize, &method, sizeof(method));
  }
  return optimizer_class;
}

static void *optimizer_ctor(void *self, __attribute__((unused)) va_list *arg) {
  struct xab_optimizer *optimizer = self;
  return optimizer;
}

static void *optimizer_dtor(void *self) {
  struct xab_optimizer *optimizer = self;
  super_dtor(XABOptimizer, self);
  return optimizer;
}

static void optimize(__attribute__((unused)) void *self,
                     __attribute__((unused)) void **weights) {
  UbjectError.error("Not optimizing?");
  return;
}

void xab_optimize(void *optimizer, void **weights, uint64_t no_of_weights) {
  const struct xab_optimizer_class *class = classOf(optimizer);

  struct xab_weight **_weights = (struct xab_weight **)weights;
  void *trainable_weights[no_of_weights];
  uint64_t no_of_trainable_weights = 0;
  for (uint64_t i = 0; i < no_of_weights; i++) {
    if (_weights[i]->trainable) {
      trainable_weights[no_of_trainable_weights++] = weights[i];
    }
  }
  if (no_of_trainable_weights) {
    class->optimize(optimizer, trainable_weights, no_of_trainable_weights);
  }
}
