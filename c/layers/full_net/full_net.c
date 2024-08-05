#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>

#include <Ubject.h>
#include <laud.h>

#define XAB_LAYER_PROTECTED
#include "../base_layer/layer.r.h"
#include "../full_net/full_net.h"

struct xab_full_net_class {
  struct xab_layer_class _;
};

struct xab_full_net {
  struct xab_layer _;
  void *input_variable;
};

struct xab_full_net_configuration {
  struct xab_layer_configuration _;
  void *weight;
  void *bias;
};

#define WEIGHT(fullnet) (weights(fullnet)[0])
#define BIAS(fullnet) (weights(fullnet)[1])

#define FULL_NET_PRIORITY (XAB_LAYER_PRIORITY + 1)

static void *fullnet_dtor(void *self);
static uint64_t fullnet_puto(void *self, FILE *f);
static void *fullnet_rollb(void *self, FILE *f);
static void build_fullnet(void *self);
static inline size_t xab_invalid_in_dim() {
  UbjectError.error("invalid input dimension (0)");
  return 0;
}
static void fullnet_change_input(void *fullnet_instance, void *new_input);

static void fini_fullnet();
const void *XABFullNet = NULL;
const void *XABFullNetClass = NULL;
static void __attribute__((constructor(FULL_NET_PRIORITY)))
init_full_net(void) {

  if (!XABFullNetClass) {
    XABFullNetClass = XABLayerClass;
  }

  if (!XABFullNet) {
    XABFullNet = init(XABFullNetClass, XABLayer, sizeof(struct xab_full_net), //
                      className, "XABFullNet",                                //
                      dtor, fullnet_dtor,                                     //
                      puto, fullnet_puto,                                     //
                      rollback, fullnet_rollb,                                //
                      xab_build_layer, build_fullnet,                         //
                      xab_set_layer_input_variable, fullnet_change_input,     //
                      xab_layer_config_size,
                      sizeof(struct xab_full_net_configuration), //
                      NULL);
  }
  atexit(fini_fullnet);
}

static void fini_fullnet() { FREE((void *)XABFullNet); }

static void *fullnet_dtor(void *self) {
  struct xab_full_net *layer = self;

  struct xab_full_net_configuration *config_info = xab_configuration(layer);

  if (config_info->weight)
    blip(config_info->weight);
  if (config_info->bias)
    blip(config_info->bias);

  config_info->weight = NULL;
  config_info->bias = NULL;

  if (layer->input_variable) {
    blip(layer->input_variable);
  }

  return super_dtor(XABFullNet, self);
}

void *xab_full_net(int n_outdim, ...) {

  va_list args;
  va_start(args, n_outdim);
  void *fullnet = init(XABFullNet, n_outdim, &args, NULL);
  va_end(args);
  return fullnet;
}

static void build_fullnet(void *full_net_instance) {

  struct xab_full_net *fullnet = full_net_instance;

  struct xab_full_net_configuration *config_info =
      xab_configuration(full_net_instance);

  uint64_t shape_of_weights[] = {config_info->_.in_dim,
                                 xab_output_dim(fullnet)};

  if (config_info->_.in_layer) {

    fullnet->input_variable =
        xab_output_computation_node(config_info->_.in_layer);
    shape_of_weights[0] = xab_output_dim(config_info->_.in_layer);
    // reference this object since it was created outside the layer
    reference(fullnet->input_variable);
  } else {
    fullnet->input_variable = laud_var();
  }

  void *Weight = laud_var();
  void *bias = laud_var();

  void *weight_value = NULL;
  void *bias_value = NULL;

  if (shape_of_weights[0] <= 0) {
    xab_invalid_in_dim();
  }

  initialize_layer_weight_nodes((struct xab_layer *)fullnet, 2);
  set_layer_weight_node((struct xab_layer *)fullnet, 0, Weight, 1);
  set_layer_weight_node((struct xab_layer *)fullnet, 1, bias, 0);

  // initialize weight
  laud_set_variable_value(
      Weight,
      weight_value = config_info->weight
                         ? config_info->weight
                         : laud_from_function(NULL, 2, shape_of_weights, NULL),
      NULL);

  // initialize bias
  laud_set_variable_value(
      bias,
      bias_value = config_info->bias ? config_info->bias
                                     : laud_from_function(
                                           NULL, 1, shape_of_weights + 1, NULL),
      NULL);

  //  laud_set_variable_value() automatically references the narray value. thus
  //  we are blip()ing the variable that may have been created here to prevent
  //  leak
  if (!config_info->weight)
    blip(weight_value);
  else
    // Since config_info.weight is now part of full net layer but not referenced
    // by config_info, we're only going to set it's holder to NULL
    config_info->weight = NULL;

  if (!config_info->bias)
    blip(bias_value);
  else
    config_info->bias = NULL;

  void *xW = laud_matrix_dot(fullnet->input_variable, Weight);
  void *xW_b = laud_add(xW, bias);
  void *act_xW_b = xab_activator(xW_b, config_info->_.activator, NULL);

  xab_set_output_computation_node(fullnet, act_xW_b);

  blip(xW);
  blip(xW_b);
}

static uint64_t fullnet_puto(void *self, FILE *f) {
  struct xab_full_net *fullnet = self;
  uint64_t len = super_puto(XABFullNet, self, f);

  uint8_t trainable;

  trainable = get_layer_weight_trainable((struct xab_layer *)fullnet, 0);
  fwrite(&trainable, sizeof(trainable), 1, f);

  puto(laud_value(get_layer_weight_node((struct xab_layer *)fullnet, 0)), f);

  trainable = get_layer_weight_trainable((struct xab_layer *)fullnet, 1);
  fwrite(&trainable, sizeof(trainable), 1, f);

  puto(laud_value(get_layer_weight_node((struct xab_layer *)fullnet, 1)), f);
  return len;
}
static void *fullnet_rollb(void *self, FILE *f) {
  struct xab_full_net *fullnet = super_rollback(XABFullNet, self, f);

  uint8_t trainable;

  fread(&trainable, sizeof(trainable), 1, f);
  void *weight = rollback(f);
  xab_full_net_set_weight(fullnet, weight);

  fread(&trainable, 1, sizeof(trainable), f);
  void *bias = rollback(f);
  xab_full_net_set_bias(fullnet, bias);

  return fullnet;
}

static void fullnet_change_input(void *fullnet_instance, void *new_input) {
  struct xab_full_net *fullnet = fullnet_instance;

  laud_set_variable_value(fullnet->input_variable, new_input, NULL);
}

void xab_full_net_set_weight(void *full_net_instance, void *weight) {
  if (xab_is_built(full_net_instance)) {
    laud_set_variable_value(get_layer_weight_node(full_net_instance, 0), weight,
                            NULL);
  } else {
    struct xab_full_net_configuration *config_info =
        xab_configuration(full_net_instance);
    config_info->weight = weight;
  }
}
void xab_full_net_set_bias(void *full_net_instance, void *bias) {
  if (xab_is_built(full_net_instance)) {
    laud_set_variable_value(get_layer_weight_node(full_net_instance, 1), bias,
                            NULL);
  } else {
    struct xab_full_net_configuration *config_info =
        xab_configuration(full_net_instance);
    config_info->bias = bias;
  }
}
void *xab_full_net_get_bias(const void *full_net_instance) {

  if (xab_is_built(full_net_instance)) {
    return get_layer_weight_node(full_net_instance, 1);
  } else {
    struct xab_full_net_configuration *config_info =
        xab_configuration(full_net_instance);
    return config_info->bias;
  }
}
