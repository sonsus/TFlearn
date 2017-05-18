#filename string is passed as tokens
filename_queue=tf.train.string_input_producer(['data1.csv', 'data2.csv', 'data3.csv'], 
                                                shuffle=False, name='filename_queue')

#string is read
reader = tf.TextLineReader()
key,val=reader.read(filename_queue)


record_defaults=[[0.],[0.],[0.],[0.]] # entry datatype initialize
xy=tf.decode_csv(val, record_defaults=record_defaults) #to decoder: how to read


