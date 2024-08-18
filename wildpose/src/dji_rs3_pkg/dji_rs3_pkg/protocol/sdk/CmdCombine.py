#encoding:utf-8
from protocol.sdk.SDKCRC import calc_crc

Seq_Init_Data = 0x0002

# 获取sequence序列
def seq_num():
	"""
	通过全局变量 SeqInit_Data 来获取下一个的 sequence number
	"""
	global Seq_Init_Data
	if Seq_Init_Data >= 0xFFFD:
		Seq_Init_Data = 0x0002
	Seq_Init_Data += 1
	seq_str = "%04x" % Seq_Init_Data
	return seq_str[2:] + ":" + seq_str[0:2]


def combine(cmd_type, cmd_set, cmd_id, data):
	"""
	拼接OSDK命令
	data为空， 则 data = ""
	"""
	if data == "":
		tmp_cmd = "{prefix}" + ":{cmd_set}:{cmd_id}".format(cmd_set=cmd_set, cmd_id=cmd_id)
	else:
		tmp_cmd = "{prefix}" + ":{cmd_set}:{cmd_id}:{data}".format(cmd_set=cmd_set, cmd_id=cmd_id,data=data)

	cmd_length = len(tmp_cmd.split(":")) + 15 #总的数据包长度
	seqnum = seq_num()

	cmd_prefix="AA" + ":" + ("%04x"%(cmd_length))[2:4] + ":" + ("%04x"%(cmd_length))[0:2] \
			   + ":{cmd_type}:{enc:02x}:{res1:02x}:{res2:02x}:{res3:02x}:{seqnum}".format(cmd_type=cmd_type, enc=0x00,res1=0x00, res2=0x00, res3=0x00,seqnum=seqnum)
	#计算CRC16
	crc16_val = calc_crc(cmd_prefix, 16)
	cmd_prefix = cmd_prefix + ":" + crc16_val

	cmd = tmp_cmd.format(prefix=cmd_prefix)
	crc32_val = calc_crc(cmd, 32)
	cmd = cmd + ":" + crc32_val

	return cmd
