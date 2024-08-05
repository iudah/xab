#include <Ubject.h>
#include <Ubject.r.h>
#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include <laud_nn.h>
#include <time.h>
#include <xab_defines.h>

#define XAB_OPTIMIZER_PROTECTED
#define XAB_SGD_PRIVATE
#include "../../layers/base_layer/layer.r.h"
#include "../../optimizers/sgd/sgd.h"
#include "../../optimizers/sgd/sgd.r.h"

const void *XABSGDClass = 0;
const void *XABSGD = 0;

static void *sgd_ctor(void *sgd, va_list *arg);

static void *sgd_dtor(void *self);

static void optimize(void *self, void **weights, uint64_t no_of_weights);

static void fini_sgd();

static void __attribute__((constructor(XAB_SGD_PRIORITY))) init_sgd(void) {
  if (!XABSGDClass) {
    XABSGDClass = XABOptimizerClass;
  }

  if (!XABSGD) {
    XABSGD = init(XABSGDClass, Ubject, sizeof(struct xab_sgd), //
                  className, "XABStocGradDesc",                //
                  ctor, sgd_ctor,                              //
                  dtor, sgd_dtor,                              //
                  xab_optimize, optimize,                      //
                  NULL);
  }

  atexit(fini_sgd);
}

static void fini_sgd() { FREE((void *)XABSGD); }

static void *sgd_ctor(void *self, va_list *arg) {
  struct xab_sgd *sgd = self;

  sgd->learning_rate = (number_t)va_arg(*arg, double);

  return sgd;
}

static void *sgd_dtor(void *self) {
  struct xab_sgd *sgd = self;
  super_dtor(XABSGD, self);
  return sgd;
}

void *xab_sgd(number_t learning_rate) {
  return init(XABSGD, learning_rate, NULL);
}

static void update_x(number_t *result, uint64_t result_length,
                     number_t **operands_vals,
                     laud_get_bc_value_fn_t get_bc_value, void *args) {
  struct xab_sgd *sgd = args;
  number_t lr = sgd->learning_rate;

  number_t values[2];
  if (get_bc_value) {

    for (uint64_t i = 0; i < result_length; i++) {

      get_bc_value(i, values);

      number_t g = values[0];
      number_t x = values[1];
      result[i] = x - lr * g;
    }
  } else {
    number_t *g = operands_vals[0];
    number_t *x = operands_vals[1];

    for (uint64_t i = 0; i < result_length; i++) {
      result[i] = x[i] - lr * g[i];
    }
  }
}

static void optimize(void *self, void **weights, uint64_t n_weights) {

  struct xab_sgd *sgd = self;

  for (uint64_t i = 0; i < n_weights; i++) {
    struct xab_weight *weight = weights[i];

    void *g = laud_derivative_of(weight->computation_node);
    void *x = laud_value(weight->computation_node);
    void *operands[] = {g, x};

    void *new_x = laud_user_elementary_fn(update_x, 2, operands, sgd);
    laud_set_variable_value(weight->computation_node, new_x, NULL);
    // we don't need to keep reference to new_x as laud_set_variable_value()
    // keeps this refernce
    blip(new_x);
  }
  return;
}
