#include <stdlib.h>
#include <sys/cdefs.h>

#include <Ubject.h>
#include <mem_lk.h>

#include "../model/model.r.h"
#include "../sequential/sequential.h"

#define XAB_SEQUENTIAL_PRIORITY (XAB_MODEL_PRIORITY + 3)

struct xab_sequential {
  struct xab_model _;
};

struct xab_sequential_class {
  struct xab_model_class _;
};

static void fini_seq();
const void *XABSequentialClass = NULL;
const void *XABSequential = NULL;

static void __attribute__((constructor(XAB_SEQUENTIAL_PRIORITY)))
init_sequential_unit(void) {

  if (!XABSequentialClass) {
    XABSequentialClass = XABModelClass;
  }

  if (!XABSequential) {
    XABSequential =
        init(XABSequentialClass, XABModel, sizeof(struct xab_sequential),
             className, "XABSequential", NULL);
  }
  atexit(fini_seq);
}

static void fini_seq() { FREE((void *)XABSequential); }

void *xab_sequential() { return init(XABSequential, 0, NULL); }
