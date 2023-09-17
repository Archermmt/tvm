/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/contrib/msc/core/printer/cpp_printer.cc
 */

#include "cpp_printer.h"

namespace tvm {
namespace contrib {
namespace msc {

void CppPrinter::PrintTypedDoc(const LiteralDoc& doc) {
  const ObjectRef& value = doc->value;
  bool defined = false;
  if (!value.defined()) {
    output_ << "nullptr";
    defined = true;
  } else if (const auto* int_imm = value.as<IntImmNode>()) {
    if (int_imm->dtype.is_bool()) {
      output_ << (int_imm->value ? "true" : "false");
      defined = true;
    }
  }
  if (!defined) {
    MSCBasePrinter::PrintTypedDoc(doc);
  }
}

void CppPrinter::PrintTypedDoc(const AttrAccessDoc& doc) {
  PrintDoc(doc->value, false);
  output_ << "->" << doc->name;
}

void CppPrinter::PrintTypedDoc(const CallDoc& doc) {
  PrintDoc(doc->callee, false);
  output_ << "(";
  PrintJoinedDocs(doc->args);
  ICHECK_EQ(doc->kwargs_keys.size(), doc->kwargs_values.size())
      << "CallDoc should have equal number of elements in kwargs_keys and kwargs_values.";
  if (doc->args.size() > 0 && doc->kwargs_keys.size() > 0) {
    output_ << ", ";
  }
  PrintJoinedDocs(doc->kwargs_values);
  output_ << ")";
}

void CppPrinter::PrintTypedDoc(const AssignDoc& doc) {
  ICHECK(doc->annotation.defined()) << "annotation should be given for assign";
  ICHECK(doc->lhs.defined()) << "lhs should be given for assign";
  PrintDoc(doc->annotation.value(), false);
  output_ << " ";
  PrintDoc(doc->lhs, false);
  if (doc->rhs.defined()) {
    output_ << " = ";
    PrintDoc(doc->rhs.value(), false);
  }
}

void CppPrinter::PrintTypedDoc(const IfDoc& doc) {
  MaybePrintComment(doc, true);
  output_ << "if (";
  PrintDoc(doc->predicate, false);
  output_ << ") {";
  PrintIndentedBlock(doc->then_branch);
  if (!doc->else_branch.empty()) {
    NewLine();
    output_ << "} else {";
    PrintIndentedBlock(doc->else_branch);
  }
  output_ << "}";
}

void CppPrinter::PrintTypedDoc(const ScopeDoc& doc) {
  MaybePrintComment(doc, true);
  ICHECK(doc->rhs.defined()) << "rhs should be given for scope";
  PrintDoc(doc->rhs);
  PrintIndentedBlock(doc->body);
}

void CppPrinter::PrintTypedDoc(const FunctionDoc& doc) {
  MaybePrintComment(doc, true);
  for (const AssignDoc& arg_doc : doc->args) {
    ICHECK(arg_doc->comment == nullptr) << "Function arg cannot have comment attached to them.";
  }
  ICHECK(doc->return_type.defined()) << "return_type should be given for function";
  PrintDoc(doc->return_type.value(), false);
  output_ << " ";
  PrintDoc(doc->name, false);
  output_ << "(";
  PrintJoinedDocs(doc->args, ", ");
  output_ << ")";
  if (doc->body.size() > 0) {
    output_ << " {";
    PrintIndentedBlock(doc->body);
    output_ << "}";
  } else {
    output_ << ";";
  }
  NewLine(false);
}

void CppPrinter::PrintTypedDoc(const ClassDoc& doc) {
  MaybePrintComment(doc, true);
  output_ << "class ";
  PrintDoc(doc->name, false);
  output_ << " {";
  for (const StmtDoc& d : doc->body) {
    PrintDoc(d);
  }
  NewLine(false);
  output_ << "};";
}

void CppPrinter::PrintTypedDoc(const CommentDoc& doc) {
  if (doc->comment.defined()) {
    output_ << "// " << doc->comment.value();
  }
}

void CppPrinter::PrintIndentedBlock(const Array<StmtDoc>& docs) {
  IncreaseIndent();
  for (const StmtDoc& d : docs) {
    PrintDoc(d);
  }
  DecreaseIndent();
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
