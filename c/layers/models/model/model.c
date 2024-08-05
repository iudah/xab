#include "mem_lk.h"
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include <Ubject.h>
#include <laud.h>
#include <xab_defines.h>

#define XAB_LAYER_PROTECTED
#include "../../../layers/loss/loss_fn.r.h"
#include "../../../layers/models/model/model.h"
#include "../../../layers/models/model/model.r.h"
#include "../../../optimizers/adam/adam.h"
#include "../../../optimizers/optimizer/optimizer.r.h"
#include "../../../optimizers/sgd/sgd.h"

struct winsize terminal;

const void *XABModel = 0;
const void *XABModelClass = 0;

static inline void terminal_change();

static void *model_class_ctor(void *self_, va_list *arg);

static void *model_ctor(void *self, va_list *arg);
static void *model_dtor(void *self);
static uint64_t model_puto(void *self, FILE *f);
static void *model_rollback(void *self, FILE *f);
static void build_model(void *self);
static void set_input_var(void *model, const void *var);
static void fini_model();

static void __attribute__((constructor(XAB_MODEL_PRIORITY))) init_model(void) {

  if (!XABModelClass) {
    XABModelClass =
        init(TypeClass, XABLayerClass, sizeof(struct xab_model_class), ctor,
             model_class_ctor, NULL);
  }

  if (!XABModel) {
    XABModel = init(XABModelClass, XABLayer, sizeof(struct xab_model), //
                    className, "XABModel",                             //
                    ctor, model_ctor,                                  //
                    dtor, model_dtor,                                  //
                    puto, model_puto,                                  //
                    rollback, model_rollback,                          //
                    xab_build_layer, build_model,                      //
                    xab_set_layer_input_variable, set_input_var,       //
                    NULL);
  }

  signal(SIGWINCH, terminal_change);
  terminal_change();

  atexit(fini_model);
}

static void fini_model() {
  FREE((void *)XABModel);
  FREE((void *)XABModelClass);
}

static void *model_class_ctor(void *self_, va_list *arg) {
  struct xab_model_class *self = super_ctor(XABModelClass, self_, arg);
  return self;
}

static void *model_ctor(void *self, va_list *arg) {
  struct xab_model *this = super_ctor(XABModel, self, arg);
  this->array_of_layers = CALLOC(2, sizeof(void *));
  this->limit = 2;
  this->n_layers = 0;
  return this;
}

static void *model_dtor(void *self) {
  struct xab_model *this = self;

  uint64_t no_of_weights = get_number_of_weights((void *)this);
  for (uint64_t i = 0; i < no_of_weights; i++) {
    set_layer_weight((struct xab_layer *)this, i, NULL);
  }

  for (uint64_t i = 0; i < this->n_layers; i++) {
    blip(this->array_of_layers[i]);
  }
  FREE(this->array_of_layers);

  blip(this->training_data->optimizer);
  FREE(this->training_data);

  super_dtor(XABModel, self);

  return this;
}

static uint64_t model_puto(void *self, FILE *f) {
  struct xab_model *this = self;
  uint64_t len = super_puto(XABModel, self, f);

  len += fwrite(&this->n_layers, 1, sizeof(this->n_layers), f);
  for (uint64_t i = 0; i < this->n_layers; i++) {
    len += puto(this->array_of_layers[i], f);
  }

  len += puto(this->training_data->optimizer, f);
  uint32_t loss_id = xab_get_loss_id(this->training_data->loss);
  len += fwrite(&loss_id, sizeof(loss_id), 1, f);
  len += fwrite(&this->training_data->accuracy,
                sizeof(this->training_data->accuracy), 1, f);

  return len;
}

static void *model_rollback(void *self, FILE *f) {
  struct xab_model *this = super_rollback(XABModel, self, f);

  fread(&this->limit, 1, sizeof(this->limit), f);

  this->array_of_layers = MALLOC(this->limit * sizeof(void *));

  for (uint64_t i = 0; i < this->limit; i++) {
    void *layer = rollback(f);
    xab_model_add(this, layer);
  }

  xab_configure(this, NULL);

  this->training_data->optimizer = rollback(f);

  uint32_t loss_id;
  fread(&loss_id, sizeof(loss_id), 1, f);
  this->training_data->loss = xab_get_loss_fn(loss_id);

  fread(&this->training_data->accuracy, sizeof(this->training_data->accuracy),
        1, f);

  return this;
}

static void build_model(void *self) {
  struct xab_model *this = self;

  uint64_t no_of_weights = 0;
  uint64_t i = 0;
  while (i < this->n_layers) {
    xab_build_layer(this->array_of_layers[i]);
    no_of_weights += get_number_of_weights(this->array_of_layers[i]);
    i++;
  }

  initialize_layer_weight_nodes((struct xab_layer *)this, no_of_weights);

  while (i) {
    i--;
    uint64_t j = get_number_of_weights(this->array_of_layers[i]);
    while (j) {
      j--;
      no_of_weights--;
      set_layer_weight((struct xab_layer *)this, no_of_weights,
                       get_layer_weight(this->array_of_layers[i], j));
    }
  }
}

void xab_model_add(void *model_instance, const void *layer) {
  struct xab_model *model = model_instance;

  if (model->n_layers > 0) {

    void *last_layer = model->array_of_layers[model->n_layers - 1];

    if (xab_is_built(layer)) {

      UbjectError.error("xab_model_add: built layer cannot be added");

    } else {
      xab_set_input_layer((void *)layer, last_layer);
    }
  }

  if (model->n_layers == model->limit) {
    model->limit *= 2;
    model->array_of_layers =
        REALLOC(model->array_of_layers, model->limit * sizeof(XABLayer));
    if (!model->array_of_layers) {
      UbjectError.error("xab_model_add: Insufficient Memory");
    }
  }

  model->array_of_layers[model->n_layers] = (struct xab_layer *)layer;
  reference(model->array_of_layers[model->n_layers]);
  model->n_layers++;
}

void xab_configure(void *model, ...) {

  struct xab_model *this = model;
  xab_build_layer(this);

  if (!this->training_data) {
    this->training_data = CALLOC(1, sizeof(struct xab_model_process));
  }

  int arg_name;

  va_list arg;
  va_start(arg, model);
  while ((arg_name = va_arg(arg, int))) {
    // turn off bit 6 or stay off
    switch (arg_name & ~XAB_PTR_ARG) {

    case XAB_LOSS: {
      this->training_data->loss = xab_get_loss_fn(va_arg(arg, int));
    } break;

    case XAB_OPTIMIZER:
      switch (va_arg(arg, int)) {
      case XAB_ADAM:
        this->training_data->optimizer = xab_adam(0.001, 0.9, 0.999, 1e-8);
        break;
      case XAB_SGD:
        this->training_data->optimizer = xab_sgd(0.5);
        break;
      default:
        UbjectError.error("loss_fn: unknown loss function");
      }
      break;

    case XAB_METRIC_START: {
      while ((arg_name = va_arg(arg, int)) != XAB_METRIC_END) {

        switch (arg_name) {

        case XAB_ACCURACY:
          this->training_data->accuracy = 1;
          break;

        default:
          UbjectError.warn("xab_configure: unknown metric\n");
          break;
        }
      }
    } break;

    default:
      UbjectError.warn("xab_configure: unknown argument\n");
      va_arg(arg, int);
    }
  }
}

char progress_bar_is_active = 0;

static inline void prep_progress_bar() {
  return;
  progress_bar_is_active = 1;

  printf("\n"
         "\033[s"
         "\033[1;%ir"
         "\033[u"
         "\033[A",
         terminal.ws_row - 1);
}
static inline void finish_progress_bar() {
  progress_bar_is_active = 0;
  printf("\033[r\033[%i;1H\n", terminal.ws_row);
}

static inline void terminal_change() {
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &terminal);
  if (progress_bar_is_active) {
    prep_progress_bar();
  }
}

void xab_model_fit(void *model_instance, void *input_vars,
                   void *expected_output, uint64_t n_epochs,
                   uint64_t batch_size) {

  struct xab_model *model = model_instance;

  uint64_t n_batch = laud_shape(input_vars)[0];
  uint64_t n_steps = n_batch > batch_size ? n_batch / batch_size : 1;

  // prepare terminal for progess bar
  prep_progress_bar();

#define PRE_NOTE "Epoch %" PRIu64 "|["
#define POST_NOTE "]|loss:             |      ns\n"
#define POST_NOTE_LEN (sizeof(POST_NOTE))
#define PROGRESS_LEN (terminal.ws_col - pre_note_len - POST_NOTE_LEN)

  char progress_buffer[terminal.ws_col];
  int16_t pre_note_len =
      snprintf(progress_buffer, terminal.ws_col, PRE_NOTE, n_epochs);
  int16_t epoch_len = pre_note_len - 8;

  snprintf(progress_buffer + pre_note_len + PROGRESS_LEN, POST_NOTE_LEN,
           POST_NOTE);

  memset(progress_buffer + pre_note_len, '.', PROGRESS_LEN);

  char buffer[64];

  // retrieve model output node
  void *output_node =
      xab_output_computation_node(model->array_of_layers[model->n_layers - 1]);

  // retrieve model loss function
  void *loss_fn_node = model->training_data->loss;

  // attach loss node
  xab_attach_loss_node(loss_fn_node, output_node);

  for (uint64_t i = 0; i < n_epochs; i++) {

    snprintf(progress_buffer + 6, epoch_len + 1, "%*" PRIu64, epoch_len, i + 1);
    progress_buffer[6 + epoch_len] = '|';
    memset(progress_buffer + pre_note_len, '#',
           PROGRESS_LEN * (i + 1) / n_epochs);

    // Todo: implement shuffling of training set

    // shuffle
    void *shuffled_input_vars = input_vars;

    void *shuffled_expected_output = expected_output;

    // start:end:step:stride
    void *generator = n_steps > 1
                          ? laud_slice_generator(shuffled_input_vars,
                                                 "0:%" PRIu64 ":1:%" PRIu64,
                                                 batch_size, batch_size)
                          : NULL;
    void *generator_eo = n_steps > 1
                             ? laud_slice_generator(shuffled_expected_output,
                                                    "0:%" PRIu64 ":1:%" PRIu64,
                                                    batch_size, batch_size)
                             : NULL;
    for (uint64_t j = 0; j < n_steps; j++) {

      time_t start_time;
      time(&start_time);

      const void *step_input =
          generator ? laud_yield_slice(generator) : shuffled_input_vars;

      const void *step_output = generator_eo ? laud_yield_slice(generator_eo)
                                             : shuffled_expected_output;

      // set input
      xab_set_layer_input_variable(model->array_of_layers[0], step_input);

      // set expected output
      xab_set_loss_objective(loss_fn_node, (void *)step_output);

      //  feed forward and compute loss
      laud_evaluate(loss_fn_node);

      // feed back
      laud_differentiate(loss_fn_node, NULL);

      //  update weights
      uint64_t no_of_weights = get_number_of_weights((struct xab_layer *)model);
      xab_optimize(model->training_data->optimizer,
                   get_layer_weights((void *)model), no_of_weights);

      time_t end_time;
      time(&end_time);

      // laud_value()
      snprintf(progress_buffer + pre_note_len + PROGRESS_LEN + 7, 14, "%.8f",
               laud_value_at_offset(loss_fn_node, 0));

      progress_buffer[pre_note_len + PROGRESS_LEN + 7 +
                      strlen(progress_buffer + pre_note_len + PROGRESS_LEN +
                             7)] = '|';

      laud_to_string(loss_fn_node, buffer, 64);

      printf("\033[s\033[%i;1H%s\033[u", terminal.ws_row, progress_buffer);
    }

    if (shuffled_input_vars != input_vars)
      blip(shuffled_input_vars);
    if (shuffled_expected_output != expected_output)
      blip(shuffled_expected_output);

    if (generator)
      blip(generator);
    if (generator_eo)
      blip(generator_eo);
  }

  xab_detach_loss_node(loss_fn_node);

  finish_progress_bar();
}

void *xab_model_predict(void *model_instance, void *x, ...) {
  struct xab_model *model = model_instance;
  void *output_node =
      xab_output_computation_node(model->array_of_layers[model->n_layers - 1]);
  xab_set_layer_input_variable(model, x);

  //  feed forward
  laud_evaluate(output_node);
  return laud_value(output_node);
}

void set_input_var(void *model_instance, const void *var) {
  struct xab_model *model = model_instance;

  return xab_set_layer_input_variable(model->array_of_layers[0], var);
}
