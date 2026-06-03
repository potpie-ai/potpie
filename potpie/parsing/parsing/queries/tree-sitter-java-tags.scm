; Methods
(method_declaration
  name: (identifier) @name.definition.method)

; Method invocations
(method_invocation
  name: (identifier) @name.reference.method)

; Class definitions
(class_declaration
  name: (identifier) @name.definition.class)

; Interface definitions
(interface_declaration
  name: (identifier) @name.definition.interface)

; Field declarations
(field_declaration
  declarator: (variable_declarator
    name: (identifier) @name.definition.field))

; Variable declarations
(local_variable_declaration
  declarator: (variable_declarator
    name: (identifier) @name.definition.variable))

; Method parameters
(formal_parameter
  name: (identifier) @name.definition.parameter)

; References to types
(type_identifier) @name.reference.type

; References to variables and fields
(identifier) @name.reference.variable
