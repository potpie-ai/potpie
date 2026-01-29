(class_declaration
 name: (identifier) @name.definition.class
 ) @definition.class

(class_declaration
  (base_list (_) @name.reference.class)
 ) @reference.class

(struct_declaration
 name: (identifier) @name.definition.class
 ) @definition.class

(struct_declaration
  (base_list (_) @name.reference.class)
 ) @reference.class

(record_declaration
 name: (identifier) @name.definition.class
 ) @definition.class

(record_declaration
  (base_list (_) @name.reference.class)
 ) @reference.class

(enum_declaration
 name: (identifier) @name.definition.class
 ) @definition.class

(interface_declaration
 name: (identifier) @name.definition.interface
 ) @definition.interface

(interface_declaration
  (base_list (_) @name.reference.interface)
 ) @reference.interface

(method_declaration
 name: (identifier) @name.definition.method
 ) @definition.method

(constructor_declaration
 name: (identifier) @name.definition.method
 ) @definition.method

(object_creation_expression
 type: (identifier) @name.reference.class
 ) @reference.class

(type_parameter_constraint
 type: (identifier) @name.reference.class
 ) @reference.class

(variable_declaration
 type: (identifier) @name.reference.class
 ) @reference.class

(invocation_expression
 function:
  (member_access_expression
    name: (identifier) @name.reference.send
 )
) @reference.send

(namespace_declaration
 name: (identifier) @name.definition.module
) @definition.module

(file_scoped_namespace_declaration
 name: (identifier) @name.definition.module
) @definition.module
