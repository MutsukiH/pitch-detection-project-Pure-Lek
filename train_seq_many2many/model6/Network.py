# ------------------------------------------------------------------------------------------------
# improt library ที่ใช้ทั้หมด
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from NoteDataset import n_note, NoteDataset
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# โครงโมเดลฝั่ง encoder 
# ------------------------------------------------------------------------------------------------

class EncoderRNN(nn.Module):
    # กำหนดโครงสร้างโดยรับพารามิเตอร์ input_size และ hidden_size เพื่อกำหนดขนาดของชั้น input และ hidden state ของ GRU
    def __init__(self, input_size, hidden_size): 
        super(EncoderRNN, self).__init__() # เรียก __init__() ของ superclass 
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True) 
        # การสร้างเลเยอร์ GRU (Gated Recurrent Unit) โดยการกำหนดพารามิเตอร์ดังนี้:
            # input_size: คือขนาดของข้อมูลนำเข้าในแต่ละชั้นของ GRU
            # hidden_size: คือขนาดของ hidden layer ใน GRU
            # batch_first=True: ระบุว่าข้อมูลที่ส่งเข้ามาใน GRU จะมีมิติแรกเป็น batch_size

    def forward(self, input):
        # ไม่ embedded เพราะมักจะทำเมื่อเกี่ยวข้องกับการทำนายข้อมูลที่มีลักษณะแบบหมวดหมู่หรือเรียงลำดับ เช่น คำหรือประโยคใน NLP, หมวดหมู่ของรูปภาพ
        output, hidden = self.gru(input) # ใช้วิธีการสร้างแบบ GRU (Gated Recurrent Unit) 
        return output, hidden

# ------------------------------------------------------------------------------------------------
# โครงโมเดลฝั่ง encoder 
# ------------------------------------------------------------------------------------------------

class DecoderRNN(nn.Module):
    # กำหนดโครงสร้างโดยรับพารามิเตอร์ hidden_size และ output_size เพื่อกำหนดขนาดของชั้น output และ hidden state ของ GRU
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__() # เรียก __init__() ของ superclass 
        self.output_size = output_size

        self.gru = nn.GRU(14, hidden_size, batch_first=True) # ใช้ 14 เพราะว่าเป็น [0,1,0,...,0] ที่ยาว 14 
        self.out = nn.Linear(hidden_size, output_size) # ใช้ในการแปลงผลลัพธ์ที่ได้จาก GRU ให้อยู่ในรูปของ output ที่ต้องการ

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # batch_size = encoder_outputs.size(0)
        decoder_input = target_tensor[:, 0] # เอา sos จาก target ที่ทำเป็น [0,1,0,...,0] มาใช้เป็นตัวเริ่ม
        decoder_hidden = encoder_hidden # เอา hidden ที่ได้จากการทำ encoder มาใช้ต่อ
        decoder_outputs = [] # สร้างไว้เก็บผลลัพท์จากการทำนาย
        MAX_LENGTH = target_tensor.size(1) # กำหนดความยาวสูงสุดของแต่ละชุดข้อมูล

        for i in range(1, MAX_LENGTH+1):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output) # เพื่อตัวที่ทำนายออกมาเข้าไปใน list 
            
            # สร้าง [0, 0, ..., 0] ที่ยาว 14 ตัวขึ้นมาใหม่
            decoder_input = torch.zeros(decoder_input.size(0), 1, self.output_size)
            for b in range(decoder_input.size(0)):
                each_decoder_output = decoder_output[b]
                _, topi = each_decoder_output.topk(1) # เอาตำแหน่งของค่าที่มากที่สุดออกมา
                # print(topi)
                if topi == n_note - 1: # ถ้าผลเป็น eos 
                    break # ให้ออกจากลูป
                else: # ถ้าไม่เป็น eos 
                    decoder_input[b][0][topi] = 1 # ให้เปลี่ยนตำแหน่งนั้นใน [0, 0, ..., 0] เป็น 1 เพื่อเป็น input ในรอบถัดไป
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1) # Tensor ทั้งหมดในลิสต์ decoder_outputs ถูกรวมกันในมิติที่ 1 ตามแนวแกนขวาง ได้เป็นขนาด 1, จำนวนตัวรวม eos, 14
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1) # ทำให้ค่า relu มาปรับค่าให้ไม่โดดมากเกินไป ให้ได้ค่าที่สามารถนำมาใช้ในการคำนวณ Loss Function ได้เหมาะสมโดยไม่เกิดปัญหา
        # return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        output = F.relu(input) # เอา [0, ..., 1, ..., 0] ของตัวก่อนหน้ามาผ่าน relu ซึ่งเป็น Activation Function ถ้า Input เป็นบวก Slope จะเท่ากับ 1 ตลอด ทำให้ Gradient ไม่หาย (ไม่เกิด Vanishing Gradient) ทำให้เทรนโมเดลได้เร็วขึ้น 
        output, hidden = self.gru(output, hidden) # เอาสิ่งที่ได้จาก relu มาผ่าน gru ที่สร้างไว้
        output = self.out(output) # นำเอา output ที่ได้มาผ่านฟัก์ชั้น สinear เพื่อ map สิ่งที่โมเดลทำนายออกมา
        return output, hidden

# ------------------------------------------------------------------------------------------------
