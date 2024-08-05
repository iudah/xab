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

#define XAB_LAYER_PRIVATE
#include "layer.r.h"

const void *XABLayerClass = 0;
const void *XABLayer = 0;

static void *layer_class_ctor(void *layer_class, va_list *arg);

static void *layer_ctor(void *layer, va_list *arg);

static void *layer_dtor(void *self);

static uint64_t layer_puto(void *self, FILE *f);

static void *layer_rollb(void *self, FILE *f);

static void build_layer(void *layer);

static void fini_layer();

static void __attribute__((constructor(XAB_LAYER_PRIORITY))) init_layer(void) {
  if (!XABLayerClass) {
    XABLayerClass =
        init(TypeClass, TypeClass, sizeof(struct xab_layer_class), //
             ctor, layer_class_ctor,                               //
             NULL);
  }

  if (!XABLayer) {
    XABLayer =
        init(XABLayerClass, Ubject, sizeof(struct xab_layer),               //
             className, "XABLayer",                                         //
             ctor, layer_ctor,                                              //
             dtor, layer_dtor,                                              //
             puto, layer_puto,                                              //
             rollback, layer_rollb,                                         //
             xab_build_layer, build_layer,                                  //
             xab_layer_config_size, sizeof(struct xab_layer_configuration), //
             NULL);
  }

  atexit(fini_layer);
}

static void fini_layer() {
  FREE((void *)XABLayer);
  FREE((void *)XABLayerClass);
}

static void *layer_class_ctor(void *self, va_list *arg) {
  struct xab_layer_class *layer_class = super_ctor(XABLayerClass, self, arg);

  typedef void (*voidf)();

  voidf selector;
  va_list args = *arg;

  while ((selector = va_arg(args, voidf))) {
    if (selector == (voidf)xab_layer_config_size) {
      layer_class->config_size = va_arg(args, size_t);
    } else {
      voidf method = va_arg(args, voidf);

      if (selector == (voidf)xab_build_layer)
        memcpy(&layer_class->build, &method, sizeof(method));
      else if (selector == (voidf)xab_set_layer_input_variable)
        memcpy(&layer_class->set_layer_input_variable, &method, sizeof(method));
    }
  }
  return layer_class;
}

static void *layer_ctor(void *self, va_list *arg) {
  struct xab_layer *layer = self;

  const struct xab_layer_class *class = classOf(self);

  layer->configuration = CALLOC(1, class->config_size);
  layer->out_dim = va_arg(*arg, size_t);

  va_list *layer_args = va_arg(*arg, va_list *);

  if (layer_args) {
    va_list layer_sub_args;
    va_copy(layer_sub_args, *layer_args);

    int selector;
    while ((selector = va_arg(layer_sub_args, int))) {

      if (selector == XAB_INPUT_DIM) {
        layer->configuration->in_dim = va_arg(layer_sub_args, int);
      } else if (selector == XAB_ACTIVATION) {
        layer->configuration->activator =
            (void *)(uintptr_t)va_arg(layer_sub_args, int);
      } else {
        // skip
        va_arg(layer_sub_args, int);
      }
    }

    if (va_arg(*arg, void *))
      UbjectError.error("expected 2 parameters and NULL but it is probably "
                        "not your fault.\n"
                        "iudah! check your init(?Layer,...)\n");
  }

  return layer;
}

static void *layer_dtor(void *self) {
  struct xab_layer *layer = self;

  if (layer->output_computation_node) {
    blip((void *)layer->output_computation_node);
  }

  if (layer->configuration) {
    FREE(layer->configuration);
  }

  for (uint64_t i = 0; i < layer->number_of_weights; i++) {
    if (layer->weights[i]) {

      blip(layer->weights[i]->computation_node);
      FREE(layer->weights[i]);
      layer->weights[i] = NULL;
    }
  }
  if (layer->number_of_weights) {
    FREE(layer->weights);
    layer->weights = NULL;
  }

  return super_dtor(XABLayer, self);
}

static uint64_t layer_puto(void *self, FILE *f) {
  struct xab_layer *this = self;
  uint64_t len = super_puto(XABLayer, self, f);

  len += fwrite(&this->configuration->in_dim, 1,
                sizeof(this->configuration->in_dim), f);

  len += fwrite(&this->out_dim, 1, sizeof(this->out_dim), f);

  len += fwrite(&this->configuration->activator, 1,
                sizeof(this->configuration->activator), f);
  return len;
}
static void *layer_rollb(void *self, FILE *f) {
  struct xab_layer *layer = super_rollback(XABLayer, self, f);

  layer->configuration = CALLOC(xab_layer_config_size(layer), 1);

  fread(&layer->configuration->in_dim, 1, sizeof(layer->configuration->in_dim),
        f);

  fread(&layer->out_dim, 1, sizeof(layer->out_dim), f);

  fread(&layer->configuration->activator, 1,
        sizeof(layer->configuration->activator), f);

  return layer;
}

static void build_layer(void *layer) {
  UbjectError.warn("%s should override this function, void "
                   "xab_build_layer(layer)\n",
                   className(layer));
}

void xab_build_layer(void *layer) {
  const struct xab_layer_class *class = classOf(layer);
  if (((struct xab_layer *)layer)->configuration) {
    class->build(layer);

    ((struct xab_layer *)layer)->configuration->built = 1;

    return;
  }
  return;
}

char xab_is_built(const void *layer_instance) {
  return ((struct xab_layer *)layer_instance)->configuration->built == 1;
}

size_t xab_output_dim(const void *layer) {
  return ((struct xab_layer *)layer)->out_dim;
}

void xab_set_input_layer(void *layer_instance, const void *input_layer) {
  if (xab_is_built(layer_instance)) {
    UbjectError.warn("input layer of built layer cannot be set");
    return;
  }
  struct xab_layer *layer = layer_instance;
  layer->configuration->in_layer = input_layer;
}

void xab_set_layer_input_variable(void *layer, const void *input_var) {
  const struct xab_layer_class *layer_class = classOf(layer);

  layer_class->set_layer_input_variable(layer, input_var);
}

void *xab_output_computation_node(const void *layer) {
  return (void *)((struct xab_layer *)layer)->output_computation_node;
}

void *xab_configuration(const void *layer) {
  return ((struct xab_layer *)layer)->configuration;
}

void xab_set_output_computation_node(void *layer, const void *out_node) {

  if (xab_is_built(layer)) {
    UbjectError.error(
        "trying to set the output computation node of a built layer");
  }

  assert(out_node);

  ((struct xab_layer *)layer)->output_computation_node = out_node;
}

void *xab_activator(void *y_hat,
                    void *(*activator_or_activator_index)(void *y_hat,
                                                          void *args),
                    void *activator_arg) {
  switch ((uintptr_t)activator_or_activator_index) {
  case (uintptr_t)XAB_RELU:
    return laud_relu(y_hat);
    break;
  case (uintptr_t)XAB_SIGMOID:
    return laud_sigmoid(y_hat);
    break;
  case (uintptr_t)NULL:
    return y_hat;
    break;
  default:
    return activator_or_activator_index(y_hat, activator_arg);
    break;
  }
}

void xab_update_layer_weight(void *layer,
                             void (*optimizer)(void *weight_node)) {
  const struct xab_layer_class *layer_class = classOf(layer);

  layer_class->update_layer_weight(layer, optimizer);
}

size_t xab_layer_config_size(void *layer) {
  const struct xab_layer_class *class = classOf(layer);
  return class->config_size;
}
