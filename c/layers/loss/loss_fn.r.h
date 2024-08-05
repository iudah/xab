#ifndef LOSS_FN_R_H
#define LOSS_FN_R_H

#include <TypeClass.r.h>
#include <Ubject.r.h>
#include <laud.h>
#include <stdint.h>

#define XAB_LOSS_FN_PRIORITY (UBJECT_PRIORITY + 303)

void xab_prep_loss_node(void *loss_fn_node, void *output_node,
                        const void *expected_output);

void xab_attach_loss_node(void *loss_fn_node, void *output_node);

void xab_set_loss_objective(void *loss_fn_node, void *expected_output);

void xab_detach_loss_node(void *loss_fn_node);

void *xab_get_loss_fn(uint32_t id);

uint32_t xab_get_loss_id(void *loss_fn);

#endif
