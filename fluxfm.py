import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import hdf5plugin
import glob
import struct
import array
from skimage.transform import warp_polar


def tag_grab(group_root):
    tag_list = []    
    raw_tag_list = glob.glob(group_root+"/*/")
    [tag_list.append(rt.split("/")[-2]) for rt in raw_tag_list]
    return sorted(tag_list)

class RedPro:

	def __init__(self,array,pix_num):
		self.array = array
		self.avg = np.array([])
		self.qar = np.zeros(pix_num)
		self.sar = np.zeros(pix_num)
		self.data = array
		self.rfac = 0.0
		self.xmin = 50
		self.xmax = 500
		self.prominence = 1.0


	def peak_hunt(self,aga):
		peaks, properties = find_peaks(self.data, prominence=self.prominence)
		peaks_in_s = []
		for i, datap in enumerate(self.data):
			if i in peaks:
				if i in aga:
					peaks_in_s.append(self.sar[i])
		peaks_in_s = np.array(peaks_in_s)
		return peaks_in_s, properties


	def norm_pattern(self,patt):
		ymax = np.max(patt)
		ymin = np.min(patt)
		baseline = patt - ymin
		norm_baseline = baseline / ymax
		return norm_baseline


	def calc_rfac(self):
		delta = 0.0
		sum_yobs = 0.0

		norm_avg = self.norm_pattern(self.avg[:,1])		
		norm_data = self.norm_pattern(self.data)
		for x in range(len(norm_data)):
			if self.xmin < x < self.xmax:
				ddelta = (norm_data[x] - norm_avg[x])**2
				delta += ddelta
				sum_yobs += norm_avg[x]**2
		rfac = np.sqrt(delta / sum_yobs)
		#if rfac > 0.4:
		#	plt.plot(norm_data[self.xmin:self.xmax])
		#	plt.plot(norm_avg[self.xmin:self.xmax])
		#	plt.show()
		return rfac

	def proc2s(self,pix_size,cam_length,wavelength):
		for i,pix in enumerate(self.sar):
			a = np.arctan((i * pix_size) / cam_length)
			q = (4 * np.pi / wavelength) * np.sin( a / 2 )
			s = (2  / wavelength) * np.sin( a / 2 )
			self.sar[i] = s
		joint_array = np.column_stack((self.sar,self.data))
		#print(self.sar.shape)
		#print(self.data.shape)
		return joint_array
	
	def proc2q(self,pix_size,cam_length,wavelength):
		for i,pix in enumerate(self.sar):
			a = np.arctan((i * pix_size) / cam_length)
			q = ((4 * np.pi) / wavelength) * np.sin( a / 2 )
			s = (2  / wavelength) * np.sin( a / 2 )
			self.qar[i] = q
		joint_array = np.column_stack((self.qar,self.data))
		#print(self.sar.shape)
		#print(self.data.shape)
		return joint_array


class XfmHfiveDataset:

    def __init__(self):
        self.root = ''
        self.group = ''
        self.tag = ''
        self.dpath = ''
        self.apath = ''
        self.scratch = ''
        self.mfpath = ''
        self.nx = 1062
        self.ny = 1028
        self.image_center = (0,0)
        self.h5ls = []
        self.dsum = []
        self.mask = []
        self.bboxes = []
        self.circs = []
        self.pix_size = 1.0
        self.cam_length = 1.0
        self.wavelength = 1.0


    def grab_dset_members(self):
        print('<grab_dset_members> Grabbing .h5 list...')
        full = sorted(os.listdir(self.dpath))
        self.h5ls = full[:-2]
        #print('h5 list: ',self.h5ls)
        return self.h5ls  # remove the _master and _11 



    def mk_scratch(self,folder):
        """
        Checks if the run has a scratch folder and makes it if needed
        :return: updates self.scratch to new locale
        """
        scr_path = self.apath + folder + '/'
        if not os.path.exists(scr_path):
            os.makedirs(scr_path)
        print(f'scratch folder: {scr_path}')
        self.scratch = scr_path



    def write_dbin(self, path, data):
        """
        Write a .dbin file to the specified path. Slower than using npy files
        :param path: str to file location
        :param data: array to be written
        :return: nothing returned 
        """
        foal = open(path, "wb")
        fmt = '<' + 'd' * data.size
        bin_in = struct.pack(fmt, *data.flatten()[:])
        foal.write(bin_in)
        foal.close()


    def read_dbin(self, path, swapbyteorder=0):
        size = os.path.getsize(path)
        print(size)
        b = array.array('d')
        fail = open(path, 'r')
        print(fail)
        b.fromfile(fail, size / 8)
        fail.close()
        lst = b.tolist()
        output = np.array(lst).reshape(self.nx, self.ny)
        if swapbyteorder == 1:
            output  = output.newbyteorder()
        return output
        


    def gen_mask(self,image,max_lim,bboxes=False,circs=False,dump=False):
        print('<gen_mask> Generating mask...')
        self.mask = np.ones(image.shape)
        for index, pix in np.ndenumerate(image):
            if pix > max_lim:
                nx = index[0]
                ny = index[1]
                self.mask[nx,ny] = 0
            else:
                continue
        if bboxes:
            print("<gen_mask> masking from bboxes")
            print(self.bboxes)
            self.mask = np.transpose(self.mask)
            for idx, pixel in np.ndenumerate(image):
                excluded_flag = False
                for exbox in self.bboxes:
                    if exbox[0] < idx[0] < exbox[1] and exbox[2] < idx[1] < exbox[3]:
                        excluded_flag = True
                        #print(excluded_flag)
                        continue
                if excluded_flag:
                    self.mask[idx[0],idx[1]] = 0
            self.mask = np.transpose(self.mask)
        if circs:
            print("<gen_mask> masking from circs")
            print(self.circs)
            self.mask = np.transpose(self.mask)
            for idx, pixel in np.ndenumerate(image):
                excluded_flag = False
                for crc in self.circs:
                    if (idx[0] - crc[0]) * (idx[0] - crc[0]) + (idx[1] - crc[1]) * (idx[1] - crc[1]) <= crc[2]**2:
                        excluded_flag = True
                        continue
                if excluded_flag:
                    self.mask[idx[0],idx[1]] = 0
            self.mask = np.transpose(self.mask)
        if dump:
            print('<gen_mask> Writing sum to:',self.scratch+self.tag + '_sum.npy')
            np.save(self.scratch+self.tag + '_mask.npy', self.mask)
            print('<gen_mask> Writing sum to:',self.scratch+self.tag + '_sum.dbin')
            self.write_dbin(self.scratch+self.tag + '_mask.dbin', self.mask)
        return self.mask

    
    def inspect_mask(self,image):
        print('<inspect_mask> inspecting mask...')
        inspection_frame = self.mask * image
        plt.imshow(inspection_frame)
        plt.show()

        
    def inspect_unwarp(self,image):
        print('<inspect_unwarp> unwarping...')
        inspec = warp_polar(np.transpose(image), center = self.image_center, radius = 1028)
        plt.imshow(inspec)
        plt.show()


    def sum_h5s(self, limit=10, dump=False):
        """
        Sums all h5 files in a run
        :param limit: set to index at which sum will stop
        :return: numpy array containing sum of all data in run
        """
        # First check to see if the sum file has been generated before:
        prior_sum_file = self.scratch+self.tag + '_sum.npy'
        if os.path.isfile(prior_sum_file):
            print('<sum_h5s> sum file already exists...continue?')
            input('Press enter to continue...')
        print('<sum_h5s> Summing all frames...')
        sum_data = np.zeros((1062, 1028))        
        for h5 in self.h5ls[:limit]:
            print('<sum_h5s> h5:', h5)
            print('<sum_h5s> Reading:', self.dpath+h5)
            with h5py.File(self.dpath+h5) as f:
                d = np.array(f['entry/data/data'])
                sum_data += np.sum(d, 0)
            # print(f'dsum.size = {sum_data.size}')
        if dump:
            print('<sum_h5s> Writing sum to:',self.scratch+self.tag + '_sum.npy')
            np.save(self.scratch+self.tag + '_sum.npy', sum_data)
            print('<sum_h5s> Writing sum to:',self.scratch+self.tag + '_sum.dbin')
            self.write_dbin(self.scratch+self.tag + '_sum.dbin', sum_data)
        self.dsum = sum_data
        return sum_data



    def atomize_h5(self,folder,limit=10,masked=False):
        """
        Each h5 file in self.h5ls is separated out into 1000 dbin files in 
        self.tag/h5_frames/
        A manifest .txt file is written out for reading by p3padf
        :param limit: int setting the number of member .h5 files in the self.tag to expand
        :
        """
        atom_path = self.scratch+folder+'/'
        if not os.path.exists(atom_path):
            os.makedirs(atom_path)
        print(f'<atomize_h5> Atomizing to {atom_path}')
        with open(atom_path+self.tag+'_manifest.txt', 'w') as f:
            for k,h5 in enumerate(self.h5ls[:limit]):
                print(f'<atomize_h5> Atomizing {h5}...')
                with h5py.File(self.dpath+h5) as h:
                    d = np.array(h['entry/data/data'])
                    for shot in range(d.shape[0]):
                        target = f'{atom_path}{self.tag}_{k}_{shot}.dbin'
                        if shot % 10 == 0:
                            print(f'<atomize_h5> {shot}/{d.shape[0]} frames generated')
                        if masked:
                            frame = d[shot,:,:] * self.mask
                        else:
                            frame = d[shot,:,:]
                        self.write_dbin(target,frame)
                        f.write(f'{atom_path}{self.tag}_{k}_{shot}.dbin'+'\n')
        print('<atomize_h5> ...complete')
        print(f'<atomize_h5> File manifest written to {atom_path}{self.tag}_manifest.txt')



    def frm_integration(self, frame):
        """
        integrate and reduce to 1d plots
        """
        frame_polar = warp_polar(np.transpose(frame), center = self.image_center, radius = 1028)
        #plt.figure()
        #plt.imshow(frame_polar)
        #plt.show()
        integrated_frame_polar = np.sum(frame_polar, axis=0)
        return integrated_frame_polar



    def reduce_h5(self,folder='1d_profiles',limit=10,masked=True,scatter_mode='q'):
        """

        """
        reduction_path = f'{self.scratch}{folder}/'
        if not os.path.exists(reduction_path):
            os.makedirs(reduction_path)
        print(f'<reduce_h5> Reducing data to 1D profiles. Output to {reduction_path}')
        print(f'<reduce_h5> Reduction mode :{scatter_mode}')
        for k,h5 in enumerate(self.h5ls[:limit]):
            print(f'<reduce_h5> Reducing {h5}...')
            with h5py.File(self.dpath+h5) as h:
                d = np.array(h['entry/data/data'])
                for shot in range(d.shape[0]):
                    frame = d[shot,:,:] * self.mask
                    profile = self.frm_integration(frame)
                    rpro = RedPro(profile,1028)
                    if scatter_mode == 'q':
                        proc_arr = rpro.proc2q(self.pix_size,self.cam_length,self.wavelength)
                        target = f'{reduction_path}{self.tag}_{k}_{shot}_reduced_q'
                        np.save(target, profile)
                    elif scatter_mode == 's':
                        proc_arr = rpro.proc2s(self.pix_size,self.cam_length,self.wavelength)
                        target = f'{reduction_path}{self.tag}_{k}_{shot}_reduced_s'
                        np.save(target, profile)




        
