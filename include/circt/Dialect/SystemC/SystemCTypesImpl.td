//===- SystemCTypesImpl.td - SystemC type implementations --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type implementations for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

class SystemCTypeDef<string name> : TypeDef<SystemCDialect, name> { }

class SignalTypesImplBase<string name> : SystemCTypeDef<name> {
  let parameters = (ins "::mlir::Type":$baseType);
  let assemblyFormat = "`<` $baseType `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$baseType), [{
      return $_get(baseType.getContext(), baseType);
    }]>
  ];
}

// Represents the SystemC sc_in<T> type.
// Declares the systemc::InputType in C++.
def InputTypeImpl : SignalTypesImplBase<"Input"> {
  let summary = "Represents a SystemC sc_in<T> value.";
  let description = [{
    Represents the specialized SystemC port class sc_in<T> as described in
    IEEE 1666-2011 §6.8.
  }];
  let mnemonic = "in";
}

// Represents the SystemC sc_inout<T> type.
// Declares the systemc::InOutType in C++.
def InOutTypeImpl : SignalTypesImplBase<"InOut"> {
  let summary = "Represents a SystemC sc_inout<T> value.";
  let description = [{
    Represents the specialized SystemC port class sc_inout<T> as described in
    IEEE 1666-2011 §6.10.
  }];
  let mnemonic = "inout";
}

// Represents the SystemC sc_out<T> type.
// Declares the systemc::OutputType in C++.
def OutputTypeImpl : SignalTypesImplBase<"Output"> {
  let summary = "Represents a SystemC sc_out<T> value.";
  let description = [{
    Represents the specialized SystemC port class sc_out<T> as described in
    IEEE 1666-2011 §6.12.
  }];
  let mnemonic = "out";
}

// Represents the SystemC sc_signal<T> type.
// Declares the systemc::SignalType in C++.
def SignalTypeImpl : SignalTypesImplBase<"Signal"> {
  let summary = "Represents a SystemC sc_signal<T> value.";
  let description = [{
    Represents the predefined primitive channel sc_signal<T> as described in
    IEEE 1666-2011 §6.4.
  }];
  let mnemonic = "signal";
}

// A type to represent systemc::SCModuleOp instances.
// Declares the systemc::ModuleType in C++.
def ModuleTypeImpl : SystemCTypeDef<"Module"> {
  let summary = "SystemC module type";
  let description = [{
    Represents a SystemC module instantiation. Example:
    `!systemc.module<moduleName(portName1: type1, portName2: type2)>`
  }];
  let mnemonic = "module";

  let hasCustomAssemblyFormat = 1;

  let parameters = (ins
    "mlir::StringAttr":$moduleName,
    ArrayRefParameter<
      "::circt::systemc::ModuleType::PortInfo", "module ports">:$ports
  );

  let extraClassDeclaration = [{
    using PortInfo = ::circt::systemc::detail::PortInfo;
  }];
}
