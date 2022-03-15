//===- Sched.td - Sched dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Sched dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SCHED_SCHED
#define CIRCT_DIALECT_SCHED_SCHED

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"

include "circt/Dialect/Sched/SchedDialect.td"

// Base class for the operations in this dialect
class SchedOp<string mnemonic, list<Trait> traits = []> :
  Op<SchedDialect, mnemonic, traits>;

// TODO
def OperationOP : SchedOp<"operation", []>;
def OpeartorTypeOP : SchedOp<"operator_type", []>;
def DependenceOP : SchedOp<"dependence", []>;
def InstanceOp : SchedOp<"instance", []>;

#endif // CIRCT_DIALECT_SCHED_SCHED
