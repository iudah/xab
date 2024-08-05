#include <Ubject.h>
#include <Ubject.r.h>
#include <assert.h>
#include <math.h>
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
#define XAB_ADAM_PRIVATE
#include "../../layers/base_layer/layer.r.h"
#include "../../optimizers/adam/adam.h"
#include "../../optimizers/adam/adam.r.h"

const void *XABAdamClass = 0;
const void *XABAdam = 0;

static void *adam_ctor(void *adam, va_list *arg);

static void *adam_dtor(void *self);

static void optimize(void *self, void **weights, uint64_t no_of_weights);

static void fini_adam();

static void __attribute__((constructor(XAB_ADAM_PRIORITY))) init_adam(void) {
  if (!XABAdamClass) {
    XABAdamClass = XABOptimizerClass;
  }

  if (!XABAdam) {
    XABAdam = init(XABAdamClass, Ubject, sizeof(struct xab_adam), //
                   className, "XABAdam",                          //
                   ctor, adam_ctor,                               //
                   dtor, adam_dtor,                               //
                   xab_optimize, optimize,                        //
                   NULL);
  }

  atexit(fini_adam);
}

static void fini_adam() { FREE((void *)XABAdam); }

static void *adam_ctor(void *self, va_list *arg) {
  struct xab_adam *adam = self;

  adam->learning_rate = va_arg(*arg, double);
  adam->beta1 = va_arg(*arg, double);
  adam->beta2 = va_arg(*arg, double);
  adam->eps = va_arg(*arg, double);
  adam->decay = 0;

  return adam;
}

static void *adam_dtor(void *self) {
  struct xab_adam *adam = self;

  for (uint64_t i = 0; i < adam->m_length; i++) {
    blip(adam->m[i]);
    blip(adam->s[i]);
  }
  if (adam->m_length)
    FREE(adam->m);
  if (adam->m_length)
    FREE(adam->s);

  super_dtor(XABAdam, self);
  return adam;
}

void *xab_adam(number_t learning_rate, number_t beta1, number_t beta2,
               number_t epsilon) {
  return init(XABAdam, learning_rate, beta1, beta2, epsilon, NULL);
}

static number_t zero(__attribute__((unused)) const uint16_t rank,
                     __attribute__((unused)) const uint64_t *const shape,
                     __attribute__((unused)) const uint64_t _,
                     __attribute__((unused)) const void *const __) {
  return 0;
}

static void update_m(number_t *result, uint64_t result_length,
                     number_t **operands_vals,
                     laud_get_bc_value_fn_t get_bc_value, void *args) {
  struct xab_adam *adam = args;
  number_t values[2];
  if (get_bc_value) {

    for (uint64_t i = 0; i < result_length; i++) {
      get_bc_value(i, values);
      result[i] = adam->beta1 * values[0] + (1. - adam->beta1) * values[1];
    }
  } else {
    number_t *m = operands_vals[0];
    number_t *g = operands_vals[1];
    for (uint64_t i = 0; i < result_length; i++) {
      result[i] = adam->beta1 * m[i] + (1. - adam->beta1) * g[i];
    }
  }
}

static void update_s(number_t *result, uint64_t result_length,
                     number_t **operands_vals,
                     laud_get_bc_value_fn_t get_bc_value, void *args) {
  struct xab_adam *adam = args;
  number_t values[2];
  if (get_bc_value) {

    for (uint64_t i = 0; i < result_length; i++) {
      get_bc_value(i, values);
      result[i] =
          adam->beta2 * values[0] + (1. - adam->beta2) * values[1] * values[1];
    }
  } else {
    number_t *m = operands_vals[0];
    number_t *g = operands_vals[1];
    for (uint64_t i = 0; i < result_length; i++) {
      result[i] = adam->beta2 * m[i] + (1. - adam->beta2) * g[i] * g[i];
    }
  }
}

static void update_x(number_t *result, uint64_t result_length,
                     number_t **operands_vals,
                     laud_get_bc_value_fn_t get_bc_value, void *args) {
  struct xab_adam *adam = args;
  number_t alpha =
      adam->learning_rate *
      (number_t)sqrt(1. - (number_t)pow(adam->beta2, 2 * adam->decay)) /
      (1. - (number_t)pow(adam->beta1, adam->decay));
  number_t values[3];
  if (get_bc_value) {

    for (uint64_t i = 0; i < result_length; i++) {

      get_bc_value(i, values);

      number_t m = values[0];
      number_t s = values[1];
      number_t x = values[2];
      result[i] = x - alpha * m / ((number_t)sqrt(s) + adam->eps);
    }
  } else {
    number_t *m = operands_vals[0];
    number_t *s = operands_vals[1];
    number_t *x = operands_vals[2];

    for (uint64_t i = 0; i < result_length; i++) {
      result[i] = x[i] - alpha * m[i] / ((number_t)sqrt(s[i]) + adam->eps);
    }
  }
}

static void optimize(void *self, void **weights, uint64_t n_weights) {
  struct xab_adam *adam = self;
  adam->decay++;

  if (!adam->m_length) {
    adam->m_length = n_weights;
    void **m = adam->m = CALLOC(n_weights, sizeof(void *));
    void **s = adam->s = CALLOC(n_weights, sizeof(void *));

    for (uint64_t i = 0; i < n_weights; i++) {
      uint64_t one = 1;
      m[i] = laud_from_function(zero, 1, &one, NULL);
      s[i] = laud_from_function(zero, 1, &one, NULL);
    }
  }

  for (uint64_t i = 0; i < n_weights; i++) {
    struct xab_weight *weight = weights[i];

    void *g = laud_derivative_of(weight->computation_node);
    void *x = laud_value(weight->computation_node);
    void *operands[] = {adam->m[i], g, x};
    void *new_m = laud_user_elementary_fn(update_m, 2, operands, adam);
    blip(adam->m[i]);
    adam->m[i] = new_m;

    operands[0] = adam->s[i];
    void *new_s = laud_user_elementary_fn(update_s, 2, operands, adam);
    blip(adam->s[i]);
    adam->s[i] = new_s;

    operands[0] = adam->m[i];
    operands[1] = adam->s[i];
    void *new_x = laud_user_elementary_fn(update_x, 3, operands, adam);
    laud_set_variable_value(weight->computation_node, new_x, NULL);
    // we don't need to keep reference to new_x as laud_set_variable_value()
    // keeps this refernce
    blip(new_x);
  }
  return;
}
