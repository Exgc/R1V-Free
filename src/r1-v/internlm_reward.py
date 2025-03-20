import torch
from transformers import AutoModel, AutoTokenizer

internlm_path = '/mnt/private_hk/data/internlm-xcomposer2d5-7b-reward'

model = AutoModel.from_pretrained(
    internlm_path,
    # device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
print(model)
print('111', model.plora_sub_GN, model.plora_glb_GN)

tokenizer = AutoTokenizer.from_pretrained(internlm_path, trust_remote_code=True)
model.tokenizer = tokenizer
model=model.to('cuda')
model = model.eval()

chat_1 = [
    {"role": "user", "content": 'I want to buy a car from the input image, analyze the advantages and weaknesses.'},
    {"role": "assistant",
     "content": "The car in the image is a Mercedes-Benz G-Class. This is a luxury SUV known for its ruggedness, off-road capability, and iconic boxy design. Here's an analysis of its advantages and weaknesses: Advantages: Off-road capability: The G-Class is built on a robust ladder frame chassis and has a powerful engine, making it highly capable in off-road conditions. It features a low-range transfer case, locking differentials, and a high ground clearance. Luxury and comfort: Despite its rugged exterior, the G-Class offers a luxurious interior with high-quality materials, comfortable seats, and advanced technology features. Iconic design: The distinctive boxy shape of the G-Class has made it a cult classic. It's instantly recognizable and has a timeless appeal. Strong resale value: Due to its exclusivity and desirability, the G-Class tends to hold its value well over time. Weaknesses: Fuel efficiency: The G-Class is not known for its fuel efficiency, especially with the larger engine options. Ride quality: The firm suspension is designed for off-road performance, which can make the ride a bit bumpy on paved roads. Price: The G-Class is a very expensive vehicle, even for a luxury SUV. Size and maneuverability: The large size of the G-Class can make it challenging to park and maneuver in tight spaces."}
]
chat_2 = [
    {"role": "user", "content": 'I want to buy a car from the input image, analyze the advantages and weaknesses.'},
    {"role": "assistant",
     "content": "Based on the image, it appears to be a Ferrari F8 Tributo. Advantages: Performance: Ferraris are renowned for their exceptional performance. The F8 Tributo is no exception, boasting a powerful twin-turbocharged V8 engine that delivers exhilarating acceleration and top speed. Handling: Ferraris are known for their precise handling and agility. The F8 Tributo continues this tradition, with advanced aerodynamics and sophisticated suspension systems ensuring a thrilling driving experience. Design: Ferrari is synonymous with iconic and head-turning designs. The F8 Tributo is a testament to this, featuring sleek lines, aggressive styling, and a captivating presence. Exclusivity: Owning a Ferrari is a status symbol. It represents luxury, exclusivity, and a passion for high-performance automobiles. Weaknesses: Price: Ferraris come with a hefty price tag. The F8 Tributo is no exception, making it an investment for those with significant financial resources. Fuel Efficiency: High-performance sports cars like the F8 Tributo are not known for their fuel efficiency. You can expect lower miles per gallon compared to everyday vehicles. Maintenance Costs: Owning a Ferrari comes with associated maintenance costs, which can be higher than those for regular cars. Practicality: The F8 Tributo is primarily a two-seater sports car, making it less practical for everyday use or carrying passengers. Ride Comfort: While the F8 Tributo offers a thrilling driving experience, its stiff suspension might not be ideal for long-distance comfort."}
]
image = ['/mnt/private_hk/data/RLHF-V-Dataset-img/llava1.5_raw_images/00024/000241122.jpg']
hd_num = 9
with torch.no_grad():
    # get reward score for a single chat
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        score1 = model.get_score(chat_1, image, hd_num=hd_num)
        score2 = model.get_score(chat_2, image, hd_num=hd_num)
        print("score1: ", score1)
        print("score2: ", score2)
#
# # batch inference, get multiple scores at once
# with torch.autocast(device_type='cuda', dtype=torch.float16):
#     scores = model.get_scores([chat_1, chat_2], [image, image], hd_num=hd_num)
# print("scores: ", scores)
# # >>> scores:  [5.76171875, -2.845703125]


# compare whether chat_1 is better than chat_2
with torch.autocast(device_type='cuda', dtype=torch.float16):
    compare_res = model.compare(chat_1, image, chat_2, image, hd_num=hd_num)
print("compare_res: ", compare_res)
# >>> compare_res:  True


# rank multiple chats, it will return the ranking index of each chat
# the chat with the highest score will have ranking index as 0
with torch.autocast(device_type='cuda', dtype=torch.float16):
    rank_res = model.rank([chat_1, chat_2], [image, image], hd_num=hd_num)
print("rank_res: ", rank_res)  # lower index means higher score
# >>> rank_res:  [0, 1]
