//===- SchedOps.h - Declare Sched dialect operations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Sched dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SCHED_SCHEDOPS_H
#define CIRCT_DIALECT_SCHED_SCHEDOPS_H

#include "circt/Dialect/Sched/SchedDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Sched/Sched.h.inc"

#endif // CIRCT_DIALECT_SCHED_SCHEDOPS_H
