#ifndef LAYER_R_H
#define LAYER_R_H

#include <stdint.h>

#include <TypeClass.r.h>
#include <Ubject.r.h>

struct xab_layer {

#ifndef XAB_LAYER_PRIVATE
  char ___[sizeof(struct xab_layer_private {
#endif
    struct Ubject _;

#ifndef XAB_LAYER_PRIVATE
  })];
#endif

#if !defined(XAB_LAYER_PROTECTED) && !defined(XAB_LAYER_PRIVATE)
  char ____[sizeof(struct xab_layer_protected {
#endif
    const void *output_computation_node;

    struct xab_layer_configuration *configuration;
    struct xab_weight **weights;

    uint64_t out_dim;
    uint64_t number_of_weights;

#if !defined(XAB_LAYER_PROTECTED) && !defined(XAB_LAYER_PRIVATE)
  })];
#endif
};

struct xab_layer_class {

#ifndef XAB_LAYER_PRIVATE
  char ___[sizeof(struct XAB_LAYER_CLASS_ {
#endif
    struct TypeClass _;

    size_t (*count_parameters)(void *layer, size_t *trainable,
                               size_t *non_trainable);

    void (*build)(void *layer);

    void (*set_layer_input_variable)(void *self, const void *x);

    void (*update_layer_weight)(void *layer,
                                void (*optimizer)(void *weight_node));

    size_t config_size;

    // size_t (*output_shape)(void *self, size_t **shape);

#ifndef XAB_LAYER_PRIVATE
  })];
#endif
};

#define XAB_LAYER_PRIORITY (UBJECT_PRIORITY + 3)

struct xab_weight {
  void *computation_node;
  uint8_t trainable;
};

struct xab_layer_configuration {
  void *activator;
  const void *in_layer;
  uint64_t in_dim;
  char built;
};

extern const void *XABLayer;
extern const void *XABLayerClass;

#ifdef XAB_LAYER_PROTECTED
static inline void initialize_layer_weight_nodes(struct xab_layer *layer,
                                                 uint64_t proposed_length) {
  if (!layer->weights) {
    layer->weights = CALLOC(proposed_length, sizeof(struct xab_weight *));
    layer->number_of_weights = proposed_length;
  } else {
    if (layer->number_of_weights != proposed_length) {
      void *tmp = REALLOC(layer->weights,
                          proposed_length * sizeof(struct xab_weight *));
      if (!tmp) {
        FREE(layer->weights);
        UbjectError.error("Insufficient memory");
      }
      layer->weights = tmp;
      if (layer->number_of_weights < proposed_length) {
        memset(layer->weights +
                   layer->number_of_weights * sizeof(struct xab_weight *),
               0,
               (proposed_length - layer->number_of_weights) *
                   sizeof(struct xab_weight *));
      }
      layer->number_of_weights = proposed_length;
    }
  }
}

static inline uint64_t get_number_of_weights(const struct xab_layer *layer) {
  return layer->number_of_weights;
}

static inline struct xab_weight **layer_weight__(const struct xab_layer *layer,
                                                 uint64_t i) {
  if (!layer->weights[i]) {
    layer->weights[i] = CALLOC(1, sizeof(struct xab_weight));
  }
  return layer->weights + i;
}

static inline char set_layer_weight_trainable(struct xab_layer *layer,
                                              uint64_t index,
                                              uint8_t trainable) {
  return layer_weight__(layer, index)[0]->trainable = trainable;
}

static inline char get_layer_weight_trainable(struct xab_layer *layer,
                                              uint64_t index) {
  return layer_weight__(layer, index)[0]->trainable;
}

static inline char is_trainable(struct xab_weight *weight) {
  return weight->trainable;
}

static inline void *set_layer_weight_node(struct xab_layer *layer,
                                          uint64_t index, void *weight_node,
                                          char trainable) {

  struct xab_weight *weight = layer_weight__(layer, index)[0];
  weight->trainable = trainable;
  if (weight->computation_node)
    blip(weight->computation_node);
  return weight->computation_node = weight_node;
}

static inline void *get_layer_weight_node(const struct xab_layer *layer,
                                          uint64_t index) {
  return layer_weight__(layer, index)[0]->computation_node;
}

static inline void *get_layer_weights(struct xab_layer *layer) {
  return layer_weight__(layer, 0);
}

static inline struct xab_weight *set_layer_weight(struct xab_layer *layer,
                                                  uint64_t index,
                                                  struct xab_weight *weight) {

  struct xab_weight **weight_ = &layer->weights[index];
  if (*weight_) {
    if (!weight_[0]->computation_node) {
      FREE(*weight_);
    }
  }
  return weight_[0] = weight;
}

static inline struct xab_weight *get_layer_weight(struct xab_layer *layer,
                                                  uint64_t index) {
  return layer_weight__(layer, index)[0];
}
#endif

void xab_build_layer(void *layer);
char xab_is_built(const void *layer);
size_t xab_output_dim(const void *layer);
size_t xab_input_dim(const void *layer);
void xab_set_input_layer(void *layer, const void *input_layer);
void xab_set_layer_input_variable(void *layer, const void *input_var);
void xab_backprop(void *model_instance, void *output_node,
                  void *expected_output);
void *xab_output_computation_node(const void *layer);
void *xab_configuration(const void *layer);
void xab_set_output_computation_node(void *layer, const void *out_node);
void *xab_activator(void *y_hat,
                    void *(*activator_or_activator_index)(void *y_hat,
                                                          void *args),
                    void *activator_arg);
void xab_update_layer_weight(void *layer, void (*optimizer)(void *weight_node));
size_t xab_layer_config_size(void *layer);

#endif