#ifndef FULL_NET_H
#define FULL_NET_H

#include "../../core/base.h"

XABAPI void *xab_full_net(int n_outdim, ...) __attribute__((sentinel));
XABAPI void xab_full_net_set_weight(void *full_net, void *weight);
XABAPI void xab_full_net_set_bias(void *full_net, void *bias);
XABAPI void *xab_full_net_get_bias(const void *full_net);

#endif