
import torch
import torch.nn as nn
from notedataset import n_note, NoteDataset

class RNN(nn.Module): # สร้างโมเดล
    def __init__(self, input_size, hidden_size, output_size): # กำหนดขนาดของ input จำนวนโหนดใน hidden และจำนวน output ที่เป็นไปได้
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_note + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_note + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1) # รวม cate input กับ hidden เข้าด้วยกันในขั้นแรก
        # print(category.shape)
        # print(input.shape)
        # print(hidden.shape)
        # print(input_combined.shape)

        hidden = self.i2h(input_combined) # ได้ hidden จากการนำ input_combined มาผ่านการทำอะไรสักอย่างด้วย linear 
        output = self.i2o(input_combined) # ได้ output จากการนำ input_combined มาผ่านการทำอะไรสักอย่างด้วย linear 
        output_combined = torch.cat((hidden, output), 1) # เอาทั้งสองอย่างที่ได้จากการทำ linear มาคอมบายกัน
        output = self.o2o(output_combined) # เอาที่เอาสองอันนั้นมาคอมบายกันมาผ่าน linear อีกรอบ
        output = self.dropout(output) # ทำ regularization เพื่อป้องกันการ overfit 
        output = self.softmax(output) # แปลงจากความน่าจะเป็นขอทุกคลาส ให้เป็นผลลัพท์ว่าเป็นคลาสอะไร
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size) # สร้าง intial weight 
    

# ในแต่ละครั้งที่เทรน input คือ (category, ตัวหนังสือตัวปจบ, hidden state) 
# output ที่ได้คือ (ตัวหนังสือตัวถัดไป, hidden state ของรอบถัดไป)
    
# เช่น ถ้า input คือ ABCD<EOS>
#        output คือ (“A”, “B”), (“B”, “C”), (“C”, “D”), (“D”, “EOS”).
