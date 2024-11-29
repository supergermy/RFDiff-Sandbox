import json

class RecommendN:

    def __init__(self, voxel_file, unit_ang):
        self.voxel_file = voxel_file
        self.unit_ang = unit_ang


    def cal_vol_from_voxel(self, voxel_file: str, unit_ang:float) -> float:
        vol_per_voxel = unit_ang **3

        with open(voxel_file, 'r') as json_file:
            voxel_info=json.load(json_file)

        count = 0 
        voxel_sdf = voxel_info['sdfValues']


        for i in voxel_info['sdfValues']:
             if i < 0:
                count += 1
        

        total_vol = count * vol_per_voxel
        
        return total_vol    

    def suggest_range(self, vol:float) -> list:
        range_list=[]
        min_n = int((vol+5110)/167)
        range_list.append(min_n)
        max_n = int((vol+5060)/152)
        range_list.append(max_n)
        range_list.sort()
        return range_list

    def forward(self):
        volume = self.cal_vol_from_voxel(self.voxel_file, self.unit_ang)
        suggested_range=self.suggest_range(volume)
        return suggested_range

