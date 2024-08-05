#ifndef ADAM_H
#define ADAM_H

#include <Ubject.h>
#include <laud.h>

#include "../../core/base.h"

XABAPI void *xab_adam(number_t learning_rate, number_t beta1, number_t beta2,
                      number_t epsilon);

#endif
