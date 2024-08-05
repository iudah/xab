#ifndef MODEL_R_H
#define MODEL_R_H

#include "../../base_layer/layer.r.h"
#include <stdint.h>

struct xab_model {
  struct xab_layer _;
  struct xab_model_process *training_data;
  struct xab_layer **array_of_layers;
  size_t n_layers;
  size_t limit;
};

struct xab_model_class {
  struct xab_layer_class _;
};

extern const void *XABModel;
extern const void *XABModelClass;

#define XAB_MODEL_PRIORITY (XAB_LAYER_PRIORITY + 3)

struct xab_model_process {
  void *loss;
  void *optimizer;
  uint32_t accuracy;
};

#endif
