//===- SchedDialect.td - Sched dialect definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Sched dialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SCHED_SCHEDDIALECT
#define CIRCT_DIALECT_SCHED_SCHEDDIALECT

def SchedDialect : Dialect {
  let name = "sched";

  let summary = "Types and operations for Sched Dialect";
  let description = [{
    This dialect defines the `sched` dialect, which is used as
    an import/export format for scheduling problem instances.
  }];
  let cppNamespace = "::circt::sched";
}

#endif // CIRCT_DIALECT_SCHED_SCHEDDIALECT
