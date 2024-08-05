#ifndef OPTIMIZER_R_H
#define OPTIMIZER_R_H

#include <stdint.h>

#include <TypeClass.r.h>
#include <Ubject.r.h>

struct xab_optimizer {

#ifndef XAB_OPTIMIZER_PRIVATE
  char ___[sizeof(struct xab_optimizer_private {
#endif
    struct Ubject _;

#ifndef XAB_OPTIMIZER_PRIVATE
  })];
#endif

};

struct xab_optimizer_class {

#ifndef XAB_OPTIMIZER_PROTECTED
  char ___[sizeof(struct XAB_OPTIMIZER_CLASS_ {
#endif
    struct TypeClass _;

    void (*optimize)(void *optimizer, void **weights, uint64_t no_of_weights);

#ifndef XAB_OPTIMIZER_PROTECTED
  })];
#endif
};

#define XAB_OPTIMIZER_PRIORITY (UBJECT_PRIORITY + 3)

extern const void *XABOptimizer;
extern const void *XABOptimizerClass;

#ifdef XAB_OPTIMIZER_PROTECTED
#endif

void xab_optimize(void *adam, void **weights, uint64_t no_of_weights);

#endif
