?	?
F%Eɖ@?
F%Eɖ@!?
F%Eɖ@	f1??*@f1??*@!f1??*@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?
F%Eɖ@r?Md??-@A?>??h??@Y'?y?3?g@rEagerKernelExecute 0*@`??.4?@)      ?=2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor&??|IU@!?(?^ZxU@)&??|IU@1?(?^ZxU@:Preprocessing2F
Iterator::Modelt??gyF+@!娗??+@)?o{???*@1=V?j?+@:Preprocessing2U
Iterator::Model::ParallelMapV2f??S9???!é*?&??)f??S9???1é*?&??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?bE?SU@!?,??U@)?t?? ???1?>?^n??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice*???O??!/[?HYA??)*???O??1/[?HYA??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?F??`U@!?
#??U@)ӡ??n,??1???/eP??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????̿?!?C??	??)???-c??1?XY???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ɧ???!?Е(????)?PS?'|?1d??f|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 13.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9e1??*@Iә??,?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	r?Md??-@r?Md??-@!r?Md??-@      ?!       "      ?!       *      ?!       2	?>??h??@?>??h??@!?>??h??@:      ?!       B      ?!       J	'?y?3?g@'?y?3?g@!'?y?3?g@R      ?!       Z	'?y?3?g@'?y?3?g@!'?y?3?g@b      ?!       JCPU_ONLYYe1??*@b qә??,?U@Y      Y@q???}<??"?
both?Your program is MODERATELY input-bound because 13.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 