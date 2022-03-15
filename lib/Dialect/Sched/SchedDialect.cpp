//===- SchedDialect.cpp - Implement the Sched dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Sched dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sched/SchedDialect.h"
#include "circt/Dialect/Sched/SchedOps.h"
#include "mlir/IR/Builders.h"

using namespace circt;
using namespace sched;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "circt/Dialect/Sched/SchedDialect.cpp.inc"

void SchedDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Sched/Sched.cpp.inc"
      >();
}