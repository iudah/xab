#include <bits/strcasecmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <Ubject.h>
#include <laud.h>
#include <laud_nn.h>
#include <string.h>
#include <time.h>

#include "../xab_defines.h"
#include "BaseObject.h"
#include "layers/full_net/full_net.h"
#include "layers/models/model/model.h"
#include "layers/models/sequential/sequential.h"

void laud_narray_test() {

  uint64_t two_x_two[] = {2, 2};
  number_t weight_0_nos[] = {.15, .25, .20, .30};
  number_t weight_1_nos[] = {.40, .50, .45, .55};

  uint64_t one[] = {1};
  number_t bias_0_val[] = {.35};
  number_t bias_1_val[] = {.60};

  void *weight_0 = laud_narray(2, two_x_two, 4, weight_0_nos);
  void *weight_1 = laud_narray(2, two_x_two, 4, weight_1_nos);
  void *bias_0 = laud_narray(1, one, 1, bias_0_val);
  void *bias_1 = laud_narray(1, one, 1, bias_1_val);

  void *test_var_0 = laud_matrix_dot(weight_0, weight_1);
  void *test_var_1 = laud_add(test_var_0, bias_0);
  void *test_var_2 = laud_add(test_var_1, bias_1);

  void *sum = laud_value(test_var_2);

  char buffer[2048];
  laud_to_string(sum, buffer, 2048);
  puts(buffer);

  blip(test_var_2);
  blip(test_var_1);
  blip(test_var_0);
  blip(weight_0);
  blip(weight_1);
  blip(bias_0);
  blip(bias_1);
}

void laud_var_test() {

  uint64_t two_x_two[] = {2, 2};
  number_t weight_0_nos[] = {.15, .25, .20, .30};
  number_t weight_1_nos[] = {.40, .50, .45, .55};

  uint64_t one[] = {1};
  number_t bias_0_val[] = {.35};

  void *weight_0 = laud_narray(2, two_x_two, 4, weight_0_nos);
  void *weight_1 = laud_narray(2, two_x_two, 4, weight_1_nos);
  void *bias_0 = laud_narray(1, one, 1, bias_0_val);

  void *a = laud_var();
  void *b = laud_var();
  void *c = laud_var();

  laud_set_variable_value(a, weight_0, NULL);
  laud_set_variable_value(b, weight_1, NULL);
  laud_set_variable_value(c, bias_0, NULL);

  void *d = laud_matrix_dot(a, b);
  void *e = laud_add(c, d);
  void *f = laud_sigmoid(e);

  laud_evaluate(f);

  void *sum = laud_value(f);

  char buffer[2048];
  laud_to_string(sum, buffer, 2048);
  puts(buffer);

  blip(f);
  blip(e);
  blip(d);
  blip(c);
  blip(b);
  blip(a);

  blip(weight_0);
  blip(weight_1);
  blip(bias_0);
}

// See: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
void matt_mazur_example() {

  uint64_t one_x_two[] = {1, 2};
  number_t X_nos[] = {.05, .10};
  number_t Y_nos[] = {.01, .99};

  uint64_t two_x_two[] = {2, 2};
  number_t weight_0_nos[] = {.15, .25, .20, .30};
  number_t weight_1_nos[] = {.40, .50, .45, .55};

  uint64_t one[] = {1};
  number_t bias_0_val[] = {.35};
  number_t bias_1_val[] = {.60};

  void *weight_0 = laud_narray(2, two_x_two, 4, weight_0_nos);
  void *weight_1 = laud_narray(2, two_x_two, 4, weight_1_nos);
  void *bias_0 = laud_narray(1, one, 1, bias_0_val);
  void *bias_1 = laud_narray(1, one, 1, bias_1_val);

  void *X = laud_narray(2, one_x_two, 2, X_nos);
  void *Y = laud_narray(2, one_x_two, 2, Y_nos);

  void *layer_0 =
      xab_full_net(2, XAB_INPUT_DIM, 2, XAB_ACTIVATION, XAB_SIGMOID, NULL);

  void *layer_1 =
      xab_full_net(2, XAB_INPUT_DIM, 2, XAB_ACTIVATION, XAB_SIGMOID, NULL);

  // define model
  void *model = xab_sequential();

  xab_model_add(model, layer_0);
  xab_model_add(model, layer_1);

  xab_full_net_set_weight(layer_0, weight_0);
  xab_full_net_set_weight(layer_1, weight_1);

  xab_full_net_set_bias(layer_0, bias_0);
  xab_full_net_set_bias(layer_1, bias_1);

  xab_configure(model, XAB_LOSS, XAB_MSE, XAB_OPTIMIZER, XAB_SGD, NULL);

  xab_model_fit(model, X, Y, 1 // 0001
                ,
                1);

  void *pred = xab_model_predict(model, X, NULL);

  char buffer[2048];
  laud_to_string(pred, buffer, 2048);
  puts(buffer);

  serialize(model, "./matt_mazur_example.xab");
  printf("Model saved to %s\n", realpath("./matt_mazur_example.xab", buffer));

  void *saved_model = deserialize("./matt_mazur_example.xab");
  printf("Model retrieved from %s\n",
         realpath("./matt_mazur_example.xab", buffer));

  void *saved_pred = xab_model_predict(saved_model, X, NULL);

  laud_to_string(saved_pred, buffer, 2048);
  puts(buffer);

  // NB: There is no need to blip "pred"/"saved_pred", the outputs of the
  // xab_model_predict() call because they are part of their respective models.
  // blip(saved_pred);
  // blip(pred);

  blip(saved_model);

  blip(model);
  blip(layer_1);
  blip(layer_0);

  blip(bias_1);
  blip(bias_0);
  blip(weight_1);
  blip(weight_0);

  blip(X);
  blip(Y);

  return;
}

// See:
// https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
void pima_classifier_example() {
  void *dataset = laud_from_text("./pima-indians-diabetes.csv", ",");

  while (!dataset) {
    UbjectError.warn("pima-indians-diabetes.csv not found\n");
    printf("Enter path to pima-indians-diabetes.csv or \"skip\" to skip pima "
           "classifier:");
    char path[2048];
    fgets(path, sizeof(path), stdin);
    if (strlen(path) == 5 && path[4] == '\n') {
      // 5 because fgets always include the newline from stdin
      char skip[5] = "skip";
      memcpy(skip, path, 4);
      if (!strcasecmp(skip, "skip")) {
        printf("skipping pima classifier\n");
        return;
      }
    }

    dataset = laud_from_text(path, ",");
  }

  void *X = laud_slice(dataset, ":, 0:8");
  void *y = laud_slice(dataset, ":, 8");

  void *hidden_1 =
      xab_full_net(12, XAB_INPUT_DIM, 8, XAB_ACTIVATION, XAB_RELU, NULL);

  void *hidden_2 =
      xab_full_net(8, XAB_INPUT_DIM, 12, XAB_ACTIVATION, XAB_RELU, NULL);

  void *output =
      xab_full_net(1, XAB_INPUT_DIM, 8, XAB_ACTIVATION, XAB_SIGMOID, NULL);

  void *pima_classifier = xab_sequential();

  xab_model_add(pima_classifier, hidden_1);
  xab_model_add(pima_classifier, hidden_2);
  xab_model_add(pima_classifier, output);

  xab_configure(pima_classifier, XAB_LOSS, XAB_BINARY_CROSS_ENTROPY,
                XAB_OPTIMIZER, XAB_ADAM, NULL);

  xab_model_fit(pima_classifier, X, y, 1, 6400);

  void *pred = xab_model_predict(pima_classifier, X, NULL);

  char buffer[2048];
  laud_to_string(pred, buffer, 2048);
  puts(buffer);

  serialize(pima_classifier, "./pima_classifier.xab");
  printf("Model saved to %s\n", realpath("./pima_classifier.xab", buffer));

  void *pima_classifier_retrieve = deserialize("./pima_classifier.xab");
  printf("Model saved to %s\n", realpath("./pima_classifier.xab", buffer));

  pred = xab_model_predict(pima_classifier_retrieve, X, NULL);

  laud_to_string(pred, buffer, 2048);
  puts(buffer);

  blip(pima_classifier_retrieve);
  blip(pima_classifier);
  blip(output);
  blip(hidden_2);
  blip(hidden_1);

  blip(X);
  blip(y);
  blip(dataset);
}

#if !defined STANDALONE
#if defined _WIN32
__declspec(dllexport)
#endif
int main_()
#else
int main()
#endif
{

  laud_narray_test();

  laud_var_test();

  matt_mazur_example();

  pima_classifier_example();

  return 0;
}
