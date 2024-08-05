#include <Ubject.h>
#include <laud.h>
#include <laud_nn.h>
#include <stddef.h>
#include <stdlib.h>

#include <xab_defines.h>

#include "loss_fn.r.h"

const void *tmp_var = NULL;
const void *xab_binary_cross_entropy = NULL;
const void *xab_mse = NULL;

static void deinit_loss_fn(void);
static void __attribute__((constructor(XAB_LOSS_FN_PRIORITY)))
init_loss_fn(void) {
  if (!tmp_var)
    tmp_var = laud_var();
  if (!xab_binary_cross_entropy)
    xab_binary_cross_entropy =
        laud_binary_cross_entropy((void *)tmp_var, (void *)tmp_var);

  if (!xab_mse)
    xab_mse = laud_mse((void *)tmp_var, (void *)tmp_var);

  atexit(deinit_loss_fn);
}
static void deinit_loss_fn(void) {
  blip((void *)xab_mse);
  blip((void *)xab_binary_cross_entropy);
  blip((void *)tmp_var);
}

void xab_attach_loss_node(void *loss_fn_node, void *output_node) {
  laud_replace_independent_node(loss_fn_node, 0, output_node);
}
void xab_set_loss_objective(void *__attribute__((unused)) loss_fn_node,
                            void *expected_output) {
  // todo: use loss_fn_node
  laud_set_variable_value((void *)tmp_var, (void *)expected_output, NULL);
}

void xab_prep_loss_node0(void *loss_fn_node, void *output_node,
                         const void *expected_output) {
  laud_replace_independent_node(loss_fn_node, 0, output_node);
  laud_set_variable_value((void *)tmp_var, (void *)expected_output, NULL);
}

void xab_detach_loss_node(void *loss_fn_node) {
  laud_replace_independent_node(loss_fn_node, 0, (void *)tmp_var);
  laud_unset_variable_value((void *)tmp_var);
}

void *xab_get_loss_fn(uint32_t id) {
  switch (id) {
  case XAB_BINARY_CROSS_ENTROPY:
    return (void *)xab_binary_cross_entropy;
    break;
  case XAB_MSE:
    return (void *)xab_mse;
    break;
  default:
    UbjectError.error("loss_fn: unknown loss function");
    return NULL;
  }
}

uint32_t xab_get_loss_id(void *loss_fn) {

  if (loss_fn == xab_binary_cross_entropy)
    return XAB_BINARY_CROSS_ENTROPY;
  else if (loss_fn == xab_mse)
    return XAB_MSE;
  else {
    UbjectError.error("loss_fn: unknown loss function");
    return 0;
  }
}
