Here's an example of how to call a function in the tool set:
(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)

to=example_tool_name.example_function_name

<atem:function_calls>
<atem:invoke name="example_tool_name.example_function_name">
<atem:parameter name="example_parameter_1">value_1</atem:parameter>
<atem:parameter name="example_parameter_2">This is the value for the second parameter
that can span
"multiple" lines
</atem:parameter>
</atem:invoke>
</atem:function_calls>
