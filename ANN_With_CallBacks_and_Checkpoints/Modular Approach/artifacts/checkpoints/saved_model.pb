??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.02unknown8??
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
x
outputLAYER/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameoutputLAYER/bias
q
$outputLAYER/bias/Read/ReadVariableOpReadVariableOpoutputLAYER/bias*
_output_shapes
:
*
dtype0
?
outputLAYER/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*#
shared_nameoutputLAYER/kernel
y
&outputLAYER/kernel/Read/ReadVariableOpReadVariableOpoutputLAYER/kernel*
_output_shapes

:@
*
dtype0
z
hiddenLAYER4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namehiddenLAYER4/bias
s
%hiddenLAYER4/bias/Read/ReadVariableOpReadVariableOphiddenLAYER4/bias*
_output_shapes
:@*
dtype0
?
hiddenLAYER4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*$
shared_namehiddenLAYER4/kernel
|
'hiddenLAYER4/kernel/Read/ReadVariableOpReadVariableOphiddenLAYER4/kernel*
_output_shapes
:	?@*
dtype0
{
hiddenLAYER3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namehiddenLAYER3/bias
t
%hiddenLAYER3/bias/Read/ReadVariableOpReadVariableOphiddenLAYER3/bias*
_output_shapes	
:?*
dtype0
?
hiddenLAYER3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namehiddenLAYER3/kernel
}
'hiddenLAYER3/kernel/Read/ReadVariableOpReadVariableOphiddenLAYER3/kernel* 
_output_shapes
:
??*
dtype0
{
hiddenLAYER2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namehiddenLAYER2/bias
t
%hiddenLAYER2/bias/Read/ReadVariableOpReadVariableOphiddenLAYER2/bias*
_output_shapes	
:?*
dtype0
?
hiddenLAYER2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namehiddenLAYER2/kernel
}
'hiddenLAYER2/kernel/Read/ReadVariableOpReadVariableOphiddenLAYER2/kernel* 
_output_shapes
:
??*
dtype0
{
hiddenLAYER1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namehiddenLAYER1/bias
t
%hiddenLAYER1/bias/Read/ReadVariableOpReadVariableOphiddenLAYER1/bias*
_output_shapes	
:?*
dtype0
?
hiddenLAYER1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namehiddenLAYER1/kernel
}
'hiddenLAYER1/kernel/Read/ReadVariableOpReadVariableOphiddenLAYER1/kernel* 
_output_shapes
:
??*
dtype0
?
 serving_default_inputLAYER_inputPlaceholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_inputLAYER_inputhiddenLAYER1/kernelhiddenLAYER1/biashiddenLAYER2/kernelhiddenLAYER2/biashiddenLAYER3/kernelhiddenLAYER3/biashiddenLAYER4/kernelhiddenLAYER4/biasoutputLAYER/kerneloutputLAYER/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_73152

NoOpNoOp
?2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?1
value?1B?1 B?1
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
?
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias*
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias*
J
0
1
$2
%3
,4
-5
46
57
<8
=9*
J
0
1
$2
%3
,4
-5
46
57
<8
=9*
* 
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
* 
:
Kiter
	Ldecay
Mlearning_rate
Nmomentum*

Oserving_default* 
* 
* 
* 
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Utrace_0* 

Vtrace_0* 

0
1*

0
1*
* 
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

\trace_0* 

]trace_0* 
c]
VARIABLE_VALUEhiddenLAYER1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhiddenLAYER1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

ctrace_0* 

dtrace_0* 
c]
VARIABLE_VALUEhiddenLAYER2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhiddenLAYER2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 
?
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

jtrace_0* 

ktrace_0* 
c]
VARIABLE_VALUEhiddenLAYER3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhiddenLAYER3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

qtrace_0* 

rtrace_0* 
c]
VARIABLE_VALUEhiddenLAYER4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhiddenLAYER4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*
* 
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

xtrace_0* 

ytrace_0* 
b\
VARIABLE_VALUEoutputLAYER/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEoutputLAYER/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

z0
{1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
|	variables
}	keras_api
	~total
	count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

~0
1*

|	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'hiddenLAYER1/kernel/Read/ReadVariableOp%hiddenLAYER1/bias/Read/ReadVariableOp'hiddenLAYER2/kernel/Read/ReadVariableOp%hiddenLAYER2/bias/Read/ReadVariableOp'hiddenLAYER3/kernel/Read/ReadVariableOp%hiddenLAYER3/bias/Read/ReadVariableOp'hiddenLAYER4/kernel/Read/ReadVariableOp%hiddenLAYER4/bias/Read/ReadVariableOp&outputLAYER/kernel/Read/ReadVariableOp$outputLAYER/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_73472
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehiddenLAYER1/kernelhiddenLAYER1/biashiddenLAYER2/kernelhiddenLAYER2/biashiddenLAYER3/kernelhiddenLAYER3/biashiddenLAYER4/kernelhiddenLAYER4/biasoutputLAYER/kerneloutputLAYER/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotal_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_73536??
?

?
F__inference_outputLAYER_layer_call_and_return_conditional_losses_73395

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
,__inference_hiddenLAYER1_layer_call_fn_73304

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_72802p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
*__inference_sequential_layer_call_fn_73202

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_73013o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_72836

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_outputLAYER_layer_call_fn_73384

inputs
unknown:@

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_outputLAYER_layer_call_and_return_conditional_losses_72870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
,__inference_hiddenLAYER3_layer_call_fn_73344

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_72836p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_72802

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_inputLAYER_layer_call_and_return_conditional_losses_73295

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
*__inference_sequential_layer_call_fn_73061
inputlayer_input
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_73013o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_nameinputLAYER_input
?2
?
E__inference_sequential_layer_call_and_return_conditional_losses_73284

inputs?
+hiddenlayer1_matmul_readvariableop_resource:
??;
,hiddenlayer1_biasadd_readvariableop_resource:	??
+hiddenlayer2_matmul_readvariableop_resource:
??;
,hiddenlayer2_biasadd_readvariableop_resource:	??
+hiddenlayer3_matmul_readvariableop_resource:
??;
,hiddenlayer3_biasadd_readvariableop_resource:	?>
+hiddenlayer4_matmul_readvariableop_resource:	?@:
,hiddenlayer4_biasadd_readvariableop_resource:@<
*outputlayer_matmul_readvariableop_resource:@
9
+outputlayer_biasadd_readvariableop_resource:

identity??#hiddenLAYER1/BiasAdd/ReadVariableOp?"hiddenLAYER1/MatMul/ReadVariableOp?#hiddenLAYER2/BiasAdd/ReadVariableOp?"hiddenLAYER2/MatMul/ReadVariableOp?#hiddenLAYER3/BiasAdd/ReadVariableOp?"hiddenLAYER3/MatMul/ReadVariableOp?#hiddenLAYER4/BiasAdd/ReadVariableOp?"hiddenLAYER4/MatMul/ReadVariableOp?"outputLAYER/BiasAdd/ReadVariableOp?!outputLAYER/MatMul/ReadVariableOpa
inputLAYER/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   s
inputLAYER/ReshapeReshapeinputsinputLAYER/Const:output:0*
T0*(
_output_shapes
:???????????
"hiddenLAYER1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
hiddenLAYER1/MatMulMatMulinputLAYER/Reshape:output:0*hiddenLAYER1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#hiddenLAYER1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
hiddenLAYER1/BiasAddBiasAddhiddenLAYER1/MatMul:product:0+hiddenLAYER1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
hiddenLAYER1/ReluReluhiddenLAYER1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"hiddenLAYER2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
hiddenLAYER2/MatMulMatMulhiddenLAYER1/Relu:activations:0*hiddenLAYER2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#hiddenLAYER2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
hiddenLAYER2/BiasAddBiasAddhiddenLAYER2/MatMul:product:0+hiddenLAYER2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
hiddenLAYER2/ReluReluhiddenLAYER2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"hiddenLAYER3/MatMul/ReadVariableOpReadVariableOp+hiddenlayer3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
hiddenLAYER3/MatMulMatMulhiddenLAYER2/Relu:activations:0*hiddenLAYER3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#hiddenLAYER3/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
hiddenLAYER3/BiasAddBiasAddhiddenLAYER3/MatMul:product:0+hiddenLAYER3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
hiddenLAYER3/ReluReluhiddenLAYER3/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"hiddenLAYER4/MatMul/ReadVariableOpReadVariableOp+hiddenlayer4_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
hiddenLAYER4/MatMulMatMulhiddenLAYER3/Relu:activations:0*hiddenLAYER4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
#hiddenLAYER4/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
hiddenLAYER4/BiasAddBiasAddhiddenLAYER4/MatMul:product:0+hiddenLAYER4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
hiddenLAYER4/ReluReluhiddenLAYER4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
!outputLAYER/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0?
outputLAYER/MatMulMatMulhiddenLAYER4/Relu:activations:0)outputLAYER/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
"outputLAYER/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
outputLAYER/BiasAddBiasAddoutputLAYER/MatMul:product:0*outputLAYER/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
n
outputLAYER/SoftmaxSoftmaxoutputLAYER/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
l
IdentityIdentityoutputLAYER/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp$^hiddenLAYER1/BiasAdd/ReadVariableOp#^hiddenLAYER1/MatMul/ReadVariableOp$^hiddenLAYER2/BiasAdd/ReadVariableOp#^hiddenLAYER2/MatMul/ReadVariableOp$^hiddenLAYER3/BiasAdd/ReadVariableOp#^hiddenLAYER3/MatMul/ReadVariableOp$^hiddenLAYER4/BiasAdd/ReadVariableOp#^hiddenLAYER4/MatMul/ReadVariableOp#^outputLAYER/BiasAdd/ReadVariableOp"^outputLAYER/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2J
#hiddenLAYER1/BiasAdd/ReadVariableOp#hiddenLAYER1/BiasAdd/ReadVariableOp2H
"hiddenLAYER1/MatMul/ReadVariableOp"hiddenLAYER1/MatMul/ReadVariableOp2J
#hiddenLAYER2/BiasAdd/ReadVariableOp#hiddenLAYER2/BiasAdd/ReadVariableOp2H
"hiddenLAYER2/MatMul/ReadVariableOp"hiddenLAYER2/MatMul/ReadVariableOp2J
#hiddenLAYER3/BiasAdd/ReadVariableOp#hiddenLAYER3/BiasAdd/ReadVariableOp2H
"hiddenLAYER3/MatMul/ReadVariableOp"hiddenLAYER3/MatMul/ReadVariableOp2J
#hiddenLAYER4/BiasAdd/ReadVariableOp#hiddenLAYER4/BiasAdd/ReadVariableOp2H
"hiddenLAYER4/MatMul/ReadVariableOp"hiddenLAYER4/MatMul/ReadVariableOp2H
"outputLAYER/BiasAdd/ReadVariableOp"outputLAYER/BiasAdd/ReadVariableOp2F
!outputLAYER/MatMul/ReadVariableOp!outputLAYER/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
? 
?
E__inference_sequential_layer_call_and_return_conditional_losses_73121
inputlayer_input&
hiddenlayer1_73095:
??!
hiddenlayer1_73097:	?&
hiddenlayer2_73100:
??!
hiddenlayer2_73102:	?&
hiddenlayer3_73105:
??!
hiddenlayer3_73107:	?%
hiddenlayer4_73110:	?@ 
hiddenlayer4_73112:@#
outputlayer_73115:@

outputlayer_73117:

identity??$hiddenLAYER1/StatefulPartitionedCall?$hiddenLAYER2/StatefulPartitionedCall?$hiddenLAYER3/StatefulPartitionedCall?$hiddenLAYER4/StatefulPartitionedCall?#outputLAYER/StatefulPartitionedCall?
inputLAYER/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLAYER_layer_call_and_return_conditional_losses_72789?
$hiddenLAYER1/StatefulPartitionedCallStatefulPartitionedCall#inputLAYER/PartitionedCall:output:0hiddenlayer1_73095hiddenlayer1_73097*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_72802?
$hiddenLAYER2/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER1/StatefulPartitionedCall:output:0hiddenlayer2_73100hiddenlayer2_73102*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_72819?
$hiddenLAYER3/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER2/StatefulPartitionedCall:output:0hiddenlayer3_73105hiddenlayer3_73107*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_72836?
$hiddenLAYER4/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER3/StatefulPartitionedCall:output:0hiddenlayer4_73110hiddenlayer4_73112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_72853?
#outputLAYER/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER4/StatefulPartitionedCall:output:0outputlayer_73115outputlayer_73117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_outputLAYER_layer_call_and_return_conditional_losses_72870{
IdentityIdentity,outputLAYER/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp%^hiddenLAYER1/StatefulPartitionedCall%^hiddenLAYER2/StatefulPartitionedCall%^hiddenLAYER3/StatefulPartitionedCall%^hiddenLAYER4/StatefulPartitionedCall$^outputLAYER/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2L
$hiddenLAYER1/StatefulPartitionedCall$hiddenLAYER1/StatefulPartitionedCall2L
$hiddenLAYER2/StatefulPartitionedCall$hiddenLAYER2/StatefulPartitionedCall2L
$hiddenLAYER3/StatefulPartitionedCall$hiddenLAYER3/StatefulPartitionedCall2L
$hiddenLAYER4/StatefulPartitionedCall$hiddenLAYER4/StatefulPartitionedCall2J
#outputLAYER/StatefulPartitionedCall#outputLAYER/StatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_nameinputLAYER_input
?

?
*__inference_sequential_layer_call_fn_73177

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_72877o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
F
*__inference_inputLAYER_layer_call_fn_73289

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLAYER_layer_call_and_return_conditional_losses_72789a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
? 
?
E__inference_sequential_layer_call_and_return_conditional_losses_72877

inputs&
hiddenlayer1_72803:
??!
hiddenlayer1_72805:	?&
hiddenlayer2_72820:
??!
hiddenlayer2_72822:	?&
hiddenlayer3_72837:
??!
hiddenlayer3_72839:	?%
hiddenlayer4_72854:	?@ 
hiddenlayer4_72856:@#
outputlayer_72871:@

outputlayer_72873:

identity??$hiddenLAYER1/StatefulPartitionedCall?$hiddenLAYER2/StatefulPartitionedCall?$hiddenLAYER3/StatefulPartitionedCall?$hiddenLAYER4/StatefulPartitionedCall?#outputLAYER/StatefulPartitionedCall?
inputLAYER/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLAYER_layer_call_and_return_conditional_losses_72789?
$hiddenLAYER1/StatefulPartitionedCallStatefulPartitionedCall#inputLAYER/PartitionedCall:output:0hiddenlayer1_72803hiddenlayer1_72805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_72802?
$hiddenLAYER2/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER1/StatefulPartitionedCall:output:0hiddenlayer2_72820hiddenlayer2_72822*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_72819?
$hiddenLAYER3/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER2/StatefulPartitionedCall:output:0hiddenlayer3_72837hiddenlayer3_72839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_72836?
$hiddenLAYER4/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER3/StatefulPartitionedCall:output:0hiddenlayer4_72854hiddenlayer4_72856*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_72853?
#outputLAYER/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER4/StatefulPartitionedCall:output:0outputlayer_72871outputlayer_72873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_outputLAYER_layer_call_and_return_conditional_losses_72870{
IdentityIdentity,outputLAYER/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp%^hiddenLAYER1/StatefulPartitionedCall%^hiddenLAYER2/StatefulPartitionedCall%^hiddenLAYER3/StatefulPartitionedCall%^hiddenLAYER4/StatefulPartitionedCall$^outputLAYER/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2L
$hiddenLAYER1/StatefulPartitionedCall$hiddenLAYER1/StatefulPartitionedCall2L
$hiddenLAYER2/StatefulPartitionedCall$hiddenLAYER2/StatefulPartitionedCall2L
$hiddenLAYER3/StatefulPartitionedCall$hiddenLAYER3/StatefulPartitionedCall2L
$hiddenLAYER4/StatefulPartitionedCall$hiddenLAYER4/StatefulPartitionedCall2J
#outputLAYER/StatefulPartitionedCall#outputLAYER/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?<
?

 __inference__wrapped_model_72776
inputlayer_inputJ
6sequential_hiddenlayer1_matmul_readvariableop_resource:
??F
7sequential_hiddenlayer1_biasadd_readvariableop_resource:	?J
6sequential_hiddenlayer2_matmul_readvariableop_resource:
??F
7sequential_hiddenlayer2_biasadd_readvariableop_resource:	?J
6sequential_hiddenlayer3_matmul_readvariableop_resource:
??F
7sequential_hiddenlayer3_biasadd_readvariableop_resource:	?I
6sequential_hiddenlayer4_matmul_readvariableop_resource:	?@E
7sequential_hiddenlayer4_biasadd_readvariableop_resource:@G
5sequential_outputlayer_matmul_readvariableop_resource:@
D
6sequential_outputlayer_biasadd_readvariableop_resource:

identity??.sequential/hiddenLAYER1/BiasAdd/ReadVariableOp?-sequential/hiddenLAYER1/MatMul/ReadVariableOp?.sequential/hiddenLAYER2/BiasAdd/ReadVariableOp?-sequential/hiddenLAYER2/MatMul/ReadVariableOp?.sequential/hiddenLAYER3/BiasAdd/ReadVariableOp?-sequential/hiddenLAYER3/MatMul/ReadVariableOp?.sequential/hiddenLAYER4/BiasAdd/ReadVariableOp?-sequential/hiddenLAYER4/MatMul/ReadVariableOp?-sequential/outputLAYER/BiasAdd/ReadVariableOp?,sequential/outputLAYER/MatMul/ReadVariableOpl
sequential/inputLAYER/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
sequential/inputLAYER/ReshapeReshapeinputlayer_input$sequential/inputLAYER/Const:output:0*
T0*(
_output_shapes
:???????????
-sequential/hiddenLAYER1/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential/hiddenLAYER1/MatMulMatMul&sequential/inputLAYER/Reshape:output:05sequential/hiddenLAYER1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
.sequential/hiddenLAYER1/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/hiddenLAYER1/BiasAddBiasAdd(sequential/hiddenLAYER1/MatMul:product:06sequential/hiddenLAYER1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/hiddenLAYER1/ReluRelu(sequential/hiddenLAYER1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
-sequential/hiddenLAYER2/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential/hiddenLAYER2/MatMulMatMul*sequential/hiddenLAYER1/Relu:activations:05sequential/hiddenLAYER2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
.sequential/hiddenLAYER2/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/hiddenLAYER2/BiasAddBiasAdd(sequential/hiddenLAYER2/MatMul:product:06sequential/hiddenLAYER2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/hiddenLAYER2/ReluRelu(sequential/hiddenLAYER2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
-sequential/hiddenLAYER3/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential/hiddenLAYER3/MatMulMatMul*sequential/hiddenLAYER2/Relu:activations:05sequential/hiddenLAYER3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
.sequential/hiddenLAYER3/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/hiddenLAYER3/BiasAddBiasAdd(sequential/hiddenLAYER3/MatMul:product:06sequential/hiddenLAYER3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/hiddenLAYER3/ReluRelu(sequential/hiddenLAYER3/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
-sequential/hiddenLAYER4/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer4_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential/hiddenLAYER4/MatMulMatMul*sequential/hiddenLAYER3/Relu:activations:05sequential/hiddenLAYER4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
.sequential/hiddenLAYER4/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/hiddenLAYER4/BiasAddBiasAdd(sequential/hiddenLAYER4/MatMul:product:06sequential/hiddenLAYER4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential/hiddenLAYER4/ReluRelu(sequential/hiddenLAYER4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
,sequential/outputLAYER/MatMul/ReadVariableOpReadVariableOp5sequential_outputlayer_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0?
sequential/outputLAYER/MatMulMatMul*sequential/hiddenLAYER4/Relu:activations:04sequential/outputLAYER/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
-sequential/outputLAYER/BiasAdd/ReadVariableOpReadVariableOp6sequential_outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
sequential/outputLAYER/BiasAddBiasAdd'sequential/outputLAYER/MatMul:product:05sequential/outputLAYER/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
sequential/outputLAYER/SoftmaxSoftmax'sequential/outputLAYER/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
w
IdentityIdentity(sequential/outputLAYER/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp/^sequential/hiddenLAYER1/BiasAdd/ReadVariableOp.^sequential/hiddenLAYER1/MatMul/ReadVariableOp/^sequential/hiddenLAYER2/BiasAdd/ReadVariableOp.^sequential/hiddenLAYER2/MatMul/ReadVariableOp/^sequential/hiddenLAYER3/BiasAdd/ReadVariableOp.^sequential/hiddenLAYER3/MatMul/ReadVariableOp/^sequential/hiddenLAYER4/BiasAdd/ReadVariableOp.^sequential/hiddenLAYER4/MatMul/ReadVariableOp.^sequential/outputLAYER/BiasAdd/ReadVariableOp-^sequential/outputLAYER/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2`
.sequential/hiddenLAYER1/BiasAdd/ReadVariableOp.sequential/hiddenLAYER1/BiasAdd/ReadVariableOp2^
-sequential/hiddenLAYER1/MatMul/ReadVariableOp-sequential/hiddenLAYER1/MatMul/ReadVariableOp2`
.sequential/hiddenLAYER2/BiasAdd/ReadVariableOp.sequential/hiddenLAYER2/BiasAdd/ReadVariableOp2^
-sequential/hiddenLAYER2/MatMul/ReadVariableOp-sequential/hiddenLAYER2/MatMul/ReadVariableOp2`
.sequential/hiddenLAYER3/BiasAdd/ReadVariableOp.sequential/hiddenLAYER3/BiasAdd/ReadVariableOp2^
-sequential/hiddenLAYER3/MatMul/ReadVariableOp-sequential/hiddenLAYER3/MatMul/ReadVariableOp2`
.sequential/hiddenLAYER4/BiasAdd/ReadVariableOp.sequential/hiddenLAYER4/BiasAdd/ReadVariableOp2^
-sequential/hiddenLAYER4/MatMul/ReadVariableOp-sequential/hiddenLAYER4/MatMul/ReadVariableOp2^
-sequential/outputLAYER/BiasAdd/ReadVariableOp-sequential/outputLAYER/BiasAdd/ReadVariableOp2\
,sequential/outputLAYER/MatMul/ReadVariableOp,sequential/outputLAYER/MatMul/ReadVariableOp:a ]
/
_output_shapes
:?????????  
*
_user_specified_nameinputLAYER_input
?+
?
__inference__traced_save_73472
file_prefix2
.savev2_hiddenlayer1_kernel_read_readvariableop0
,savev2_hiddenlayer1_bias_read_readvariableop2
.savev2_hiddenlayer2_kernel_read_readvariableop0
,savev2_hiddenlayer2_bias_read_readvariableop2
.savev2_hiddenlayer3_kernel_read_readvariableop0
,savev2_hiddenlayer3_bias_read_readvariableop2
.savev2_hiddenlayer4_kernel_read_readvariableop0
,savev2_hiddenlayer4_bias_read_readvariableop1
-savev2_outputlayer_kernel_read_readvariableop/
+savev2_outputlayer_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_hiddenlayer1_kernel_read_readvariableop,savev2_hiddenlayer1_bias_read_readvariableop.savev2_hiddenlayer2_kernel_read_readvariableop,savev2_hiddenlayer2_bias_read_readvariableop.savev2_hiddenlayer3_kernel_read_readvariableop,savev2_hiddenlayer3_bias_read_readvariableop.savev2_hiddenlayer4_kernel_read_readvariableop,savev2_hiddenlayer4_bias_read_readvariableop-savev2_outputlayer_kernel_read_readvariableop+savev2_outputlayer_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapesp
n: :
??:?:
??:?:
??:?:	?@:@:@
:
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$	 

_output_shapes

:@
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?H
?

!__inference__traced_restore_73536
file_prefix8
$assignvariableop_hiddenlayer1_kernel:
??3
$assignvariableop_1_hiddenlayer1_bias:	?:
&assignvariableop_2_hiddenlayer2_kernel:
??3
$assignvariableop_3_hiddenlayer2_bias:	?:
&assignvariableop_4_hiddenlayer3_kernel:
??3
$assignvariableop_5_hiddenlayer3_bias:	?9
&assignvariableop_6_hiddenlayer4_kernel:	?@2
$assignvariableop_7_hiddenlayer4_bias:@7
%assignvariableop_8_outputlayer_kernel:@
1
#assignvariableop_9_outputlayer_bias:
&
assignvariableop_10_sgd_iter:	 '
assignvariableop_11_sgd_decay: /
%assignvariableop_12_sgd_learning_rate: *
 assignvariableop_13_sgd_momentum: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp$assignvariableop_hiddenlayer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_hiddenlayer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_hiddenlayer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_hiddenlayer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_hiddenlayer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_hiddenlayer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_hiddenlayer4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_hiddenlayer4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_outputlayer_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_outputlayer_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_sgd_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_sgd_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_sgd_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp assignvariableop_13_sgd_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_72819

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_inputLAYER_layer_call_and_return_conditional_losses_72789

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
? 
?
E__inference_sequential_layer_call_and_return_conditional_losses_73091
inputlayer_input&
hiddenlayer1_73065:
??!
hiddenlayer1_73067:	?&
hiddenlayer2_73070:
??!
hiddenlayer2_73072:	?&
hiddenlayer3_73075:
??!
hiddenlayer3_73077:	?%
hiddenlayer4_73080:	?@ 
hiddenlayer4_73082:@#
outputlayer_73085:@

outputlayer_73087:

identity??$hiddenLAYER1/StatefulPartitionedCall?$hiddenLAYER2/StatefulPartitionedCall?$hiddenLAYER3/StatefulPartitionedCall?$hiddenLAYER4/StatefulPartitionedCall?#outputLAYER/StatefulPartitionedCall?
inputLAYER/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLAYER_layer_call_and_return_conditional_losses_72789?
$hiddenLAYER1/StatefulPartitionedCallStatefulPartitionedCall#inputLAYER/PartitionedCall:output:0hiddenlayer1_73065hiddenlayer1_73067*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_72802?
$hiddenLAYER2/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER1/StatefulPartitionedCall:output:0hiddenlayer2_73070hiddenlayer2_73072*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_72819?
$hiddenLAYER3/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER2/StatefulPartitionedCall:output:0hiddenlayer3_73075hiddenlayer3_73077*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_72836?
$hiddenLAYER4/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER3/StatefulPartitionedCall:output:0hiddenlayer4_73080hiddenlayer4_73082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_72853?
#outputLAYER/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER4/StatefulPartitionedCall:output:0outputlayer_73085outputlayer_73087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_outputLAYER_layer_call_and_return_conditional_losses_72870{
IdentityIdentity,outputLAYER/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp%^hiddenLAYER1/StatefulPartitionedCall%^hiddenLAYER2/StatefulPartitionedCall%^hiddenLAYER3/StatefulPartitionedCall%^hiddenLAYER4/StatefulPartitionedCall$^outputLAYER/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2L
$hiddenLAYER1/StatefulPartitionedCall$hiddenLAYER1/StatefulPartitionedCall2L
$hiddenLAYER2/StatefulPartitionedCall$hiddenLAYER2/StatefulPartitionedCall2L
$hiddenLAYER3/StatefulPartitionedCall$hiddenLAYER3/StatefulPartitionedCall2L
$hiddenLAYER4/StatefulPartitionedCall$hiddenLAYER4/StatefulPartitionedCall2J
#outputLAYER/StatefulPartitionedCall#outputLAYER/StatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_nameinputLAYER_input
?

?
#__inference_signature_wrapper_73152
inputlayer_input
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_72776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_nameinputLAYER_input
?

?
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_73355

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_73335

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
E__inference_sequential_layer_call_and_return_conditional_losses_73013

inputs&
hiddenlayer1_72987:
??!
hiddenlayer1_72989:	?&
hiddenlayer2_72992:
??!
hiddenlayer2_72994:	?&
hiddenlayer3_72997:
??!
hiddenlayer3_72999:	?%
hiddenlayer4_73002:	?@ 
hiddenlayer4_73004:@#
outputlayer_73007:@

outputlayer_73009:

identity??$hiddenLAYER1/StatefulPartitionedCall?$hiddenLAYER2/StatefulPartitionedCall?$hiddenLAYER3/StatefulPartitionedCall?$hiddenLAYER4/StatefulPartitionedCall?#outputLAYER/StatefulPartitionedCall?
inputLAYER/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLAYER_layer_call_and_return_conditional_losses_72789?
$hiddenLAYER1/StatefulPartitionedCallStatefulPartitionedCall#inputLAYER/PartitionedCall:output:0hiddenlayer1_72987hiddenlayer1_72989*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_72802?
$hiddenLAYER2/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER1/StatefulPartitionedCall:output:0hiddenlayer2_72992hiddenlayer2_72994*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_72819?
$hiddenLAYER3/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER2/StatefulPartitionedCall:output:0hiddenlayer3_72997hiddenlayer3_72999*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_72836?
$hiddenLAYER4/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER3/StatefulPartitionedCall:output:0hiddenlayer4_73002hiddenlayer4_73004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_72853?
#outputLAYER/StatefulPartitionedCallStatefulPartitionedCall-hiddenLAYER4/StatefulPartitionedCall:output:0outputlayer_73007outputlayer_73009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_outputLAYER_layer_call_and_return_conditional_losses_72870{
IdentityIdentity,outputLAYER/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp%^hiddenLAYER1/StatefulPartitionedCall%^hiddenLAYER2/StatefulPartitionedCall%^hiddenLAYER3/StatefulPartitionedCall%^hiddenLAYER4/StatefulPartitionedCall$^outputLAYER/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2L
$hiddenLAYER1/StatefulPartitionedCall$hiddenLAYER1/StatefulPartitionedCall2L
$hiddenLAYER2/StatefulPartitionedCall$hiddenLAYER2/StatefulPartitionedCall2L
$hiddenLAYER3/StatefulPartitionedCall$hiddenLAYER3/StatefulPartitionedCall2L
$hiddenLAYER4/StatefulPartitionedCall$hiddenLAYER4/StatefulPartitionedCall2J
#outputLAYER/StatefulPartitionedCall#outputLAYER/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
,__inference_hiddenLAYER2_layer_call_fn_73324

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_72819p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?2
?
E__inference_sequential_layer_call_and_return_conditional_losses_73243

inputs?
+hiddenlayer1_matmul_readvariableop_resource:
??;
,hiddenlayer1_biasadd_readvariableop_resource:	??
+hiddenlayer2_matmul_readvariableop_resource:
??;
,hiddenlayer2_biasadd_readvariableop_resource:	??
+hiddenlayer3_matmul_readvariableop_resource:
??;
,hiddenlayer3_biasadd_readvariableop_resource:	?>
+hiddenlayer4_matmul_readvariableop_resource:	?@:
,hiddenlayer4_biasadd_readvariableop_resource:@<
*outputlayer_matmul_readvariableop_resource:@
9
+outputlayer_biasadd_readvariableop_resource:

identity??#hiddenLAYER1/BiasAdd/ReadVariableOp?"hiddenLAYER1/MatMul/ReadVariableOp?#hiddenLAYER2/BiasAdd/ReadVariableOp?"hiddenLAYER2/MatMul/ReadVariableOp?#hiddenLAYER3/BiasAdd/ReadVariableOp?"hiddenLAYER3/MatMul/ReadVariableOp?#hiddenLAYER4/BiasAdd/ReadVariableOp?"hiddenLAYER4/MatMul/ReadVariableOp?"outputLAYER/BiasAdd/ReadVariableOp?!outputLAYER/MatMul/ReadVariableOpa
inputLAYER/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   s
inputLAYER/ReshapeReshapeinputsinputLAYER/Const:output:0*
T0*(
_output_shapes
:???????????
"hiddenLAYER1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
hiddenLAYER1/MatMulMatMulinputLAYER/Reshape:output:0*hiddenLAYER1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#hiddenLAYER1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
hiddenLAYER1/BiasAddBiasAddhiddenLAYER1/MatMul:product:0+hiddenLAYER1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
hiddenLAYER1/ReluReluhiddenLAYER1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"hiddenLAYER2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
hiddenLAYER2/MatMulMatMulhiddenLAYER1/Relu:activations:0*hiddenLAYER2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#hiddenLAYER2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
hiddenLAYER2/BiasAddBiasAddhiddenLAYER2/MatMul:product:0+hiddenLAYER2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
hiddenLAYER2/ReluReluhiddenLAYER2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"hiddenLAYER3/MatMul/ReadVariableOpReadVariableOp+hiddenlayer3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
hiddenLAYER3/MatMulMatMulhiddenLAYER2/Relu:activations:0*hiddenLAYER3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#hiddenLAYER3/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
hiddenLAYER3/BiasAddBiasAddhiddenLAYER3/MatMul:product:0+hiddenLAYER3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
hiddenLAYER3/ReluReluhiddenLAYER3/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"hiddenLAYER4/MatMul/ReadVariableOpReadVariableOp+hiddenlayer4_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
hiddenLAYER4/MatMulMatMulhiddenLAYER3/Relu:activations:0*hiddenLAYER4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
#hiddenLAYER4/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
hiddenLAYER4/BiasAddBiasAddhiddenLAYER4/MatMul:product:0+hiddenLAYER4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
hiddenLAYER4/ReluReluhiddenLAYER4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
!outputLAYER/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0?
outputLAYER/MatMulMatMulhiddenLAYER4/Relu:activations:0)outputLAYER/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
"outputLAYER/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
outputLAYER/BiasAddBiasAddoutputLAYER/MatMul:product:0*outputLAYER/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
n
outputLAYER/SoftmaxSoftmaxoutputLAYER/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
l
IdentityIdentityoutputLAYER/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp$^hiddenLAYER1/BiasAdd/ReadVariableOp#^hiddenLAYER1/MatMul/ReadVariableOp$^hiddenLAYER2/BiasAdd/ReadVariableOp#^hiddenLAYER2/MatMul/ReadVariableOp$^hiddenLAYER3/BiasAdd/ReadVariableOp#^hiddenLAYER3/MatMul/ReadVariableOp$^hiddenLAYER4/BiasAdd/ReadVariableOp#^hiddenLAYER4/MatMul/ReadVariableOp#^outputLAYER/BiasAdd/ReadVariableOp"^outputLAYER/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2J
#hiddenLAYER1/BiasAdd/ReadVariableOp#hiddenLAYER1/BiasAdd/ReadVariableOp2H
"hiddenLAYER1/MatMul/ReadVariableOp"hiddenLAYER1/MatMul/ReadVariableOp2J
#hiddenLAYER2/BiasAdd/ReadVariableOp#hiddenLAYER2/BiasAdd/ReadVariableOp2H
"hiddenLAYER2/MatMul/ReadVariableOp"hiddenLAYER2/MatMul/ReadVariableOp2J
#hiddenLAYER3/BiasAdd/ReadVariableOp#hiddenLAYER3/BiasAdd/ReadVariableOp2H
"hiddenLAYER3/MatMul/ReadVariableOp"hiddenLAYER3/MatMul/ReadVariableOp2J
#hiddenLAYER4/BiasAdd/ReadVariableOp#hiddenLAYER4/BiasAdd/ReadVariableOp2H
"hiddenLAYER4/MatMul/ReadVariableOp"hiddenLAYER4/MatMul/ReadVariableOp2H
"outputLAYER/BiasAdd/ReadVariableOp"outputLAYER/BiasAdd/ReadVariableOp2F
!outputLAYER/MatMul/ReadVariableOp!outputLAYER/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_72853

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
*__inference_sequential_layer_call_fn_72900
inputlayer_input
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_72877o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_nameinputLAYER_input
?

?
F__inference_outputLAYER_layer_call_and_return_conditional_losses_72870

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_73375

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_73315

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_hiddenLAYER4_layer_call_fn_73364

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_72853o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
inputLAYER_inputA
"serving_default_inputLAYER_input:0?????????  ?
outputLAYER0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias"
_tf_keras_layer
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias"
_tf_keras_layer
f
0
1
$2
%3
,4
-5
46
57
<8
=9"
trackable_list_wrapper
f
0
1
$2
%3
,4
-5
46
57
<8
=9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32?
*__inference_sequential_layer_call_fn_72900
*__inference_sequential_layer_call_fn_73177
*__inference_sequential_layer_call_fn_73202
*__inference_sequential_layer_call_fn_73061?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
?
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32?
E__inference_sequential_layer_call_and_return_conditional_losses_73243
E__inference_sequential_layer_call_and_return_conditional_losses_73284
E__inference_sequential_layer_call_and_return_conditional_losses_73091
E__inference_sequential_layer_call_and_return_conditional_losses_73121?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
?B?
 __inference__wrapped_model_72776inputLAYER_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
I
Kiter
	Ldecay
Mlearning_rate
Nmomentum"
	optimizer
,
Oserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Utrace_02?
*__inference_inputLAYER_layer_call_fn_73289?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zUtrace_0
?
Vtrace_02?
E__inference_inputLAYER_layer_call_and_return_conditional_losses_73295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zVtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
\trace_02?
,__inference_hiddenLAYER1_layer_call_fn_73304?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z\trace_0
?
]trace_02?
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_73315?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z]trace_0
':%
??2hiddenLAYER1/kernel
 :?2hiddenLAYER1/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
?
ctrace_02?
,__inference_hiddenLAYER2_layer_call_fn_73324?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zctrace_0
?
dtrace_02?
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_73335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zdtrace_0
':%
??2hiddenLAYER2/kernel
 :?2hiddenLAYER2/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
jtrace_02?
,__inference_hiddenLAYER3_layer_call_fn_73344?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zjtrace_0
?
ktrace_02?
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_73355?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zktrace_0
':%
??2hiddenLAYER3/kernel
 :?2hiddenLAYER3/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
?
qtrace_02?
,__inference_hiddenLAYER4_layer_call_fn_73364?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zqtrace_0
?
rtrace_02?
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_73375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zrtrace_0
&:$	?@2hiddenLAYER4/kernel
:@2hiddenLAYER4/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?
xtrace_02?
+__inference_outputLAYER_layer_call_fn_73384?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zxtrace_0
?
ytrace_02?
F__inference_outputLAYER_layer_call_and_return_conditional_losses_73395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zytrace_0
$:"@
2outputLAYER/kernel
:
2outputLAYER/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_sequential_layer_call_fn_72900inputLAYER_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_73177inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_73202inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_73061inputLAYER_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_73243inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_73284inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_73091inputLAYER_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_73121inputLAYER_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
?B?
#__inference_signature_wrapper_73152inputLAYER_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_inputLAYER_layer_call_fn_73289inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_inputLAYER_layer_call_and_return_conditional_losses_73295inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_hiddenLAYER1_layer_call_fn_73304inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_73315inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_hiddenLAYER2_layer_call_fn_73324inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_73335inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_hiddenLAYER3_layer_call_fn_73344inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_73355inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_hiddenLAYER4_layer_call_fn_73364inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_73375inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_outputLAYER_layer_call_fn_73384inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_outputLAYER_layer_call_and_return_conditional_losses_73395inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
N
|	variables
}	keras_api
	~total
	count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
.
~0
1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper?
 __inference__wrapped_model_72776?
$%,-45<=A?>
7?4
2?/
inputLAYER_input?????????  
? "9?6
4
outputLAYER%?"
outputLAYER?????????
?
G__inference_hiddenLAYER1_layer_call_and_return_conditional_losses_73315^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_hiddenLAYER1_layer_call_fn_73304Q0?-
&?#
!?
inputs??????????
? "????????????
G__inference_hiddenLAYER2_layer_call_and_return_conditional_losses_73335^$%0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_hiddenLAYER2_layer_call_fn_73324Q$%0?-
&?#
!?
inputs??????????
? "????????????
G__inference_hiddenLAYER3_layer_call_and_return_conditional_losses_73355^,-0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_hiddenLAYER3_layer_call_fn_73344Q,-0?-
&?#
!?
inputs??????????
? "????????????
G__inference_hiddenLAYER4_layer_call_and_return_conditional_losses_73375]450?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? ?
,__inference_hiddenLAYER4_layer_call_fn_73364P450?-
&?#
!?
inputs??????????
? "??????????@?
E__inference_inputLAYER_layer_call_and_return_conditional_losses_73295a7?4
-?*
(?%
inputs?????????  
? "&?#
?
0??????????
? ?
*__inference_inputLAYER_layer_call_fn_73289T7?4
-?*
(?%
inputs?????????  
? "????????????
F__inference_outputLAYER_layer_call_and_return_conditional_losses_73395\<=/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????

? ~
+__inference_outputLAYER_layer_call_fn_73384O<=/?,
%?"
 ?
inputs?????????@
? "??????????
?
E__inference_sequential_layer_call_and_return_conditional_losses_73091~
$%,-45<=I?F
??<
2?/
inputLAYER_input?????????  
p 

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_73121~
$%,-45<=I?F
??<
2?/
inputLAYER_input?????????  
p

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_73243t
$%,-45<=??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_73284t
$%,-45<=??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????

? ?
*__inference_sequential_layer_call_fn_72900q
$%,-45<=I?F
??<
2?/
inputLAYER_input?????????  
p 

 
? "??????????
?
*__inference_sequential_layer_call_fn_73061q
$%,-45<=I?F
??<
2?/
inputLAYER_input?????????  
p

 
? "??????????
?
*__inference_sequential_layer_call_fn_73177g
$%,-45<=??<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
*__inference_sequential_layer_call_fn_73202g
$%,-45<=??<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
#__inference_signature_wrapper_73152?
$%,-45<=U?R
? 
K?H
F
inputLAYER_input2?/
inputLAYER_input?????????  "9?6
4
outputLAYER%?"
outputLAYER?????????
