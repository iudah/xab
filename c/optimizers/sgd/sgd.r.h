#ifndef SGD_R_H
#define SGD_R_H

#include <stdint.h>

#include <TypeClass.r.h>
#include <Ubject.r.h>
#include <laud.h>

#include "../optimizer/optimizer.r.h"

struct xab_sgd {

#ifndef XAB_SGD_PRIVATE
  char ___[sizeof(struct xab_sgd_private {
#endif
    struct xab_optimizer _;
    number_t learning_rate;

#ifndef XAB_SGD_PRIVATE
  })];
#endif

};

struct xab_sgd_class {

#ifndef XAB_SGD_PRIVATE
  char ___[sizeof(struct XAB_SGD_CLASS_ {
#endif
    struct xab_optimizer_class _;

#ifndef XAB_SGD_PRIVATE
  })];
#endif
};

#define XAB_SGD_PRIORITY (XAB_OPTIMIZER_PRIORITY + 3)

extern const void *XABSGD;
extern const void *XABSGDClass;

#ifdef XAB_SGD_PROTECTED
#endif

#endif
