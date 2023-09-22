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
 * \file src/contrib/msc/core/printer/msc_doc.h
 * \brief Extra docs for MSC
 */
#ifndef TVM_CONTRIB_MSC_CORE_PRINTER_MSC_DOC_H_
#define TVM_CONTRIB_MSC_CORE_PRINTER_MSC_DOC_H_

#include <tvm/script/printer/doc.h>

#include <string>

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief Doc that declare a var with type.
 *
 * \sa DeclareDoc
 */
class DeclareDocNode : public ExprDocNode {
 public:
  /*! \brief The type of the variable */
  ExprDoc type{nullptr};
  /*! \brief The variable */
  ExprDoc variable{nullptr};
  /*! \brief The init arguments for the variable. */
  Array<ExprDoc> init_args;
  /*! \brief Whether to use constructor(otherwise initializer) */
  bool use_constructor{true};

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("type", &type);
    v->Visit("variable", &variable);
    v->Visit("init_args", &init_args);
    v->Visit("use_constructor", &use_constructor);
  }

  static constexpr const char* _type_key = "script.printer.DeclareDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(DeclareDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of DeclareDocNode.
 *
 * \sa DeclareDocNode
 */
class DeclareDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of DeclareDoc.
   * \param type The type of the variable.
   * \param variable The variable.
   * \param init_args The init arguments of the variable.
   * \param use_constructor Whether to use constructor(otherwise initializer).
   */
  explicit DeclareDoc(ExprDoc type, ExprDoc variable, Array<ExprDoc> init_args,
                      bool use_constructor);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DeclareDoc, ExprDoc, DeclareDocNode);
};

/*!
 * \brief Doc that build a strict list, which check the empty.
 *
 * \sa StrictListDoc
 */
class StrictListDocNode : public ExprDocNode {
 public:
  /*! \brief The inner list doc */
  ListDoc list;
  /*! \brief Whether to allow empty */
  bool allow_empty{true};

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("list", &list);
    v->Visit("allow_empty", &allow_empty);
  }

  static constexpr const char* _type_key = "script.printer.StrictListDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(StrictListDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of StrictListDocNode.
 *
 * \sa StrictListDocNode
 */
class StrictListDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of StrictListDoc.
   * \param list The inner list doc.
   * \param allow_empty Whether to allow empty.
   */
  explicit StrictListDoc(ListDoc list, bool allow_empty);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(StrictListDoc, ExprDoc, StrictListDocNode);
};

/*!
 * \brief Doc that represents attribute access on another expression, visit via pointer.
 *
 * \sa PtrAttrAccessDoc
 */
class PtrAttrAccessDocNode : public ExprDocNode {
 public:
  /*! \brief The target expression to be accessed */
  ExprDoc value{nullptr};
  /*! \brief The attribute to be accessed */
  String name;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "script.printer.PtrAttrAccessDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(PtrAttrAccessDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of PtrAttrAccessDocNode.
 *
 * \sa PtrAttrAccessDocNode
 */
class PtrAttrAccessDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of PtrAttrAccessDoc
   * \param value The target expression of attribute access.
   * \param name The name of attribute to access.
   */
  explicit PtrAttrAccessDoc(ExprDoc value, String name);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(PtrAttrAccessDoc, ExprDoc, PtrAttrAccessDocNode);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_CORE_PRINTER_MSC_DOC_H_
