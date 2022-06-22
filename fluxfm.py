import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import hdf5plugin
import glob
import struct
import array
from skimage.transform import warp_polar
import re
import random
import scipy
import numpy.ma as ma
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator


def tag_grab(group_root):
    tag_list = []
    raw_tag_list = glob.glob(group_root + "/*/")
    [tag_list.append(rt.split("/")[-2]) for rt in raw_tag_list]
    return sorted(tag_list)


def sorted_nicely(ls):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(ls, key=alphanum_key)


def trim_to_qlims(qlimits, profile):
    # print(f'qlimits {qlimits}')
    snipped_q = []
    for qpoint in profile:
        if qlimits[0] < qpoint[0] < qlimits[1]:
            snipped_q.append(qpoint[0])
            snipped_int.append(qpoint[1])
            # print(qpoint[0])
        else:
            continue
    return np.column_stack((snipped_q, snipped_int))


class RedPro:

    def __init__(self, array, pix_num):
        self.array = array
        self.avg = np.array([])
        self.qar = np.zeros(pix_num)
        self.tthar = np.zeros(pix_num)
        self.sar = np.zeros(pix_num)
        self.data = array
        self.rfac = 0.0
        self.xmin = 50
        self.xmax = 500
        self.prominence = 0.01

    def peak_hunt(self):
        peaks, properties = scipy.signal.find_peaks(self.data, prominence=self.prominence)
        peaks_in_s = []
        print(peaks)
        for i, datap in enumerate(self.data):
            if i in peaks:
                peaks_in_s.append(self.qar[i])
        peaks_in_s = np.array(peaks_in_s)
        print(peaks_in_s)
        return peaks_in_s

    def norm_pattern(self, patt):
        ymax = np.max(patt)
        ymin = np.min(patt)
        baseline = patt - ymin
        norm_baseline = baseline / ymax
        return norm_baseline

    def calc_rfac(self):
        delta = 0.0
        sum_yobs = 0.0

        norm_avg = self.norm_pattern(self.avg[:, 1])
        norm_data = self.norm_pattern(self.data)
        for x in range(len(norm_data)):
            if self.xmin < x < self.xmax:
                ddelta = (norm_data[x] - norm_avg[x]) ** 2
                delta += ddelta
                sum_yobs += norm_avg[x] ** 2
        rfac = np.sqrt(delta / sum_yobs)
        # if rfac > 0.4:
        #	plt.plot(norm_data[self.xmin:self.xmax])
        #	plt.plot(norm_avg[self.xmin:self.xmax])
        #	plt.show()
        return rfac

    def proc2s(self, pix_size, cam_length, wavelength):
        for i, pix in enumerate(self.sar):
            a = np.arctan((i * pix_size) / cam_length)
            s = (2 / wavelength) * np.sin(a / 2)
            self.sar[i] = s
        joint_array = np.column_stack((self.sar, self.data))
        return joint_array

    def proc2q(self, pix_size, cam_length, wavelength):
        for i, pix in enumerate(self.sar):
            a = np.arctan((i * pix_size) / cam_length)
            q = ((4 * np.pi) / wavelength) * np.sin(a / 2)
            s = (2 / wavelength) * np.sin(a / 2)
            self.qar[i] = q * 1E-10
        joint_array = np.column_stack((self.qar, self.data))
        return joint_array

    def proc2tth(self, pix_size, cam_length, wavelength):
        for i, pix in enumerate(self.sar):
            a = np.arctan((i * pix_size) / cam_length)
            self.tthar[i] = np.degrees(a)
        joint_array = np.column_stack((self.tthar, self.data))
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
        self.experiment_id = ''
        self.nx = 1062
        self.ny = 1028
        self.max_lim = 4.29E9
        self.image_center = (0, 0)
        self.h5ls = []
        self.run_data_sum = []
        self.mask = []
        self.bboxes = []
        self.circs = []
        self.pix_size = 1.0
        self.cam_length = 1.0
        self.wavelength = 1.0
        self.average_profile = []
        self.rfactor_array = []
        self.tag_int_wt = []
        self.frm_indexes = []
        self.run_data_array = []
        self.run_data_sum = []
        self.run_data_avg = []

    def grab_dset_members(self):
        print('<grab_dset_members> Grabbing .h5 list...')
        full = sorted(os.listdir(self.dpath))
        self.h5ls = full[:]
        self.h5ls = [string for string in self.h5ls if 'data' in string]
        print('h5 list: ', self.h5ls)
        return self.h5ls  # remove the _master and _11 

    def mk_scratch(self, folder):
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
        fail = open(path, 'rb')
        # print(fail)
        b.fromfile(fail, size // 8)
        fail.close()
        lst = b.tolist()
        output = np.array(lst).reshape(self.nx, self.ny)
        if swapbyteorder == 1:
            output = output.newbyteorder()
        return output

    def gen_mask(self, image, max_lim, bboxes=False, circs=False, dump=True):
        print('<fluxfm.gen_mask> Generating mask...')
        self.mask = np.ones(image.shape)
        # print(f'<gen_mask>{self.mask.shape}')
        # print(f'<gen_mask>{self.dsum.shape}')
        for index, pix in np.ndenumerate(image):
            if pix > max_lim:
                nx = index[0]
                ny = index[1]
                self.mask[nx, ny] = 0
            else:
                continue
        if bboxes:
            print("<fluxfm.gen_mask> masking from bboxes")
            print(self.bboxes)
            print(f'<fluxfm.gen_mask>  image shape  {image.shape}')
            print(f'<fluxfm.gen_mask>  mask shape  {self.mask.shape}')
            self.mask = np.transpose(self.mask)
            image = np.transpose(image)
            print(f'<fluxfm.gen_mask>  image shape  {image.shape}')
            print(f'<fluxfm.gen_mask>  mask shape  {self.mask.shape}')
            for idx, pixel in np.ndenumerate(image):
                excluded_flag = False
                # print(idx)
                for exbox in self.bboxes:
                    if exbox[0] < idx[0] < exbox[1] and exbox[2] < idx[1] < exbox[3]:
                        excluded_flag = True
                        # print(excluded_flag)
                        continue
                if excluded_flag:
                    self.mask[idx[0], idx[1]] = 0
            self.mask = np.transpose(self.mask)
        if circs:
            print("<fluxfm.gen_mask> masking from circs")
            print(self.circs)
            self.mask = np.transpose(self.mask)
            for idx, pixel in np.ndenumerate(image):
                excluded_flag = False
                for crc in self.circs:
                    if (idx[0] - crc[0]) * (idx[0] - crc[0]) + (idx[1] - crc[1]) * (idx[1] - crc[1]) <= crc[2] ** 2:
                        excluded_flag = True
                        continue
                if excluded_flag:
                    self.mask[idx[0], idx[1]] = 0
            self.mask = np.transpose(self.mask)
        if dump:
            # print('<fluxfm.gen_mask> Writing sum to:', self.scratch + self.tag + '_sum.dbin')
            # print('<fluxfm.gen_mask> Writing sum to:', self.scratch + self.tag + '_sum.npy')
            # np.save(self.scratch + self.tag + '_sum.npy', self.sum)
            print(f'<fluxfm.gen_mask> Writing mask to:{self.scratch}{self.tag}_mask.npy')
            np.save(self.scratch + self.tag + '_mask.npy', self.mask)
        return self.mask

    def inspect_mask(self, image, cmin=0, cmax=1):
        print('<inspect_mask> inspecting mask...')
        inspection_frame = self.mask * image
        plt.imshow(inspection_frame)
        # plt.clim(cmin,cmax)
        plt.show()

    def inspect_unwarp(self, image, cmin=0, cmax=1):
        print('<inspect_unwarp> unwarping...')
        plt.figure()
        inspec = warp_polar(np.transpose(image), center=self.image_center, radius=1028)
        plt.imshow(inspec)
        plt.clim(cmin, cmax)
        plt.figure()
        profile = self.frm_integration(image, npt=2250)
        plt.plot(np.log10(profile))
        tth_rpro = RedPro(profile, 1024)
        proc_arr = tth_rpro.proc2tth(self.pix_size, self.cam_length, self.wavelength)
        plt.figure()
        plt.plot(proc_arr[:, 0], proc_arr[:, 1])
        target = f'{self.apath}{self.tag}_sum_reduced_tth.txt'
        np.savetxt(target, proc_arr)
        q_rpro = RedPro(profile, 1024)
        plt.figure()
        plt.title('s')
        s_arr = tth_rpro.proc2s(self.pix_size, self.cam_length, self.wavelength)
        plt.plot(s_arr[:, 0], s_arr[:, 1])
        plt.show()

        # pzero = 97.5
        # plt.axvline(x=pzero)
        # plt.axvline(x=pzero*np.sqrt(3))
        # plt.axvline(x=pzero*np.sqrt(4))
        # plt.show()

    def sum_h5s(self, limit=10, dump=False):
        """
        Sums all h5 files in a run
        :param limit: set to index at which sum will stop
        :return: numpy array containing sum of all data in run
        """
        # First check to see if the sum file has been generated before:
        prior_sum_file = self.scratch + self.tag + '_sum.npy'
        if os.path.isfile(prior_sum_file):
            print('<sum_h5s> sum file already exists...continue?')
            input('Press enter to continue...')
        print('<sum_h5s> Summing all frames...')
        sum_data = np.zeros((1062, 1028))
        for h5 in self.h5ls[:limit]:
            print('<sum_h5s> h5:', h5)
            print('<sum_h5s> Reading:', self.dpath + h5)
            with h5py.File(self.dpath + h5) as f:
                d = np.array(f['entry/data/data'])
                sum_data += np.sum(d, 0)
            # print(f'dsum.size = {sum_data.size}')
        if dump:
            print('<sum_h5s> Writing sum to:', self.scratch + self.tag + '_sum.npy')
            np.save(self.scratch + self.tag + '_sum.npy', sum_data)
            print('<sum_h5s> Writing sum to:', self.scratch + self.tag + '_sum.dbin')
            self.write_dbin(self.scratch + self.tag + '_sum.dbin', sum_data)
        self.run_data_sum = sum_data
        return sum_data

    def atomize_h5(self, folder, start=0, limit=10, masked=False, normalize=False, norm_range=[0, 10]):
        """
        Each h5 file in self.h5ls is separated out into 1000 dbin files in 
        self.tag/h5_frames/
        A manifest .txt file is written out for reading by p3padf
        :param limit: int setting the number of member .h5 files in the self.tag to expand
        :
        """
        atom_path = self.scratch + folder + '/'
        if not os.path.exists(atom_path):
            os.makedirs(atom_path)
        print(f'<atomize_h5> Atomizing to {atom_path}')
        with open(atom_path + self.tag + '_manifest.txt', 'w') as f:
            for k, h5 in enumerate(sorted_nicely(self.h5ls[start:limit])):
                print(f'<atomize_h5> Atomizing {h5}...')
                with h5py.File(self.dpath + h5) as h:
                    d = np.array(h['entry/data/data'])
                    for shot in range(d.shape[0]):
                        target = f'{atom_path}{self.tag}_{k}_{shot}.dbin'
                        if shot % 100 == 0:
                            print(f'<atomize_h5> {shot}/{d.shape[0]} frames generated')
                        if masked:
                            frame = d[shot, :, :] * self.mask
                        else:
                            frame = d[shot, :, :]
                        if normalize:
                            frame = self.normalize_frame(frame, norm_range)
                        self.write_dbin(target, frame)
                        f.write(f'{atom_path}{self.tag}_{k}_{shot}.dbin' + '\n')
        print('<atomize_h5> ...complete')
        print(f'<atomize_h5> File manifest written to {atom_path}{self.tag}_manifest.txt')

    def frm_integration(self, frame, unit="q_nm^-1", npt=2250):

        ai = AzimuthalIntegrator()
        ai.setFit2D(directDist=self.cam_length,
                    centerX=self.image_center[0],
                    centerY=self.image_center[1],
                    pixelX=self.pix_size,
                    pixelY=self.pix_size)
        ai.wavelength = self.wavelength / 1e10
        integrated_profile = ai.integrate1d(data=frame, npt=npt, unit=unit)
        print(np.array(integrated_profile).shape)
        return np.transpose(np.array(integrated_profile))

    def normalize_frame(self, img, norm_range):
        uw_img = self.frm_integration(img, npt=2250)
        # plt.plot(uw_img)
        norm_base = np.sum(uw_img[norm_range[0]:norm_range[1]])
        print(f'<normalize_frame> {norm_base}')
        norm_img = img / norm_base
        uw_norm = self.frm_integration(norm_img, npt=2250)
        # plt.plot(uw_norm)
        # plt.show()
        return norm_img

    def reduce_h5(self, folder='1d_profiles', start=0, limit=10, masked=False, scatter_mode='q', units='m'):
        """

        """
        reduction_path = f'{self.scratch}{folder}/'
        if not os.path.exists(reduction_path):
            os.makedirs(reduction_path)
        print(f'<reduce_h5> Reducing data to 1D profiles. Output to {reduction_path}')
        print(f'<reduce_h5> Reduction mode :{scatter_mode}')
        for k, h5 in enumerate(self.h5ls[start:limit]):
            print(f'<reduce_h5> Reducing {h5}...')
            with h5py.File(self.dpath + h5) as h:
                d = np.array(h['entry/data/data'])
                for n, shot in enumerate(range(d.shape[0])):
                    frame = d[shot, :, :] * self.mask
                    profile = self.frm_integration(frame)
                    # rpro = RedPro(profile,1024)
                    if scatter_mode == 'q':
                        # proc_arr = rpro.proc2q(self.pix_size,self.cam_length,self.wavelength)
                        target = f'{reduction_path}{self.tag}_{k}_{shot}_reduced_q_{units}.npy'
                        # print(proc_arr[100])
                        np.save(target, profile)
                    if n % 100 == 0:
                        print(f'{n} frames reduced')
                    # elif scatter_mode == 's':
                    #   proc_arr = rpro.proc2s(self.pix_size,self.cam_length,self.wavelength)
                    #  target = f'{reduction_path}{self.tag}_{k}_{shot}_reduced_s'
                    # np.save(target, proc_arr)
                    # elif scatter_mode == 'tth':
                    #   proc_arr = rpro.proc2tth(self.pix_size,self.cam_length,self.wavelength)
                    #  target = f'{reduction_path}{self.tag}_{k}_{shot}_reduced_tth'
                    # np.save(target, proc_arr)
                    # plt.plot(proc_arr[:,0],proc_arr[:,1])
                    # plt.show()

    def reduce_dbin(self, folder='1d_profiles', dbin_folder='norm_h5_frames', masked=True, scatter_mode='q'):
        """

        """
        reduction_path = f'{self.scratch}{folder}/'
        ensemble_peak = []
        ensemble_base = []
        ensemble_pob = []
        if not os.path.exists(reduction_path):
            os.makedirs(reduction_path)
        print(f'<reduce_dbin> Reducing data to 1D profiles. Output to {reduction_path}')
        print(f'<reduce_dbin> Reduction mode :{scatter_mode}')
        dbin_ls = sorted_nicely(glob.glob(f'{self.apath}{dbin_folder}/*.dbin'))
        for k, db in enumerate(dbin_ls):
            d = self.read_dbin(db)
            frame = d * self.mask
            profile = self.frm_integration(frame)
            rpro = RedPro(profile, 1028)
            if scatter_mode == 'q':
                proc_arr = rpro.proc2q(self.pix_size, self.cam_length, self.wavelength)
                target = f'{reduction_path}{self.tag}_{k}_reduced_q'
                np.save(target, profile)
                peak = np.sum(proc_arr[105:130, 1])
                base = np.sum(proc_arr[400:425, 1])
                print(f'{k} peak {peak} base {base}')
                ensemble_peak.append(peak)
                ensemble_base.append(base)
                ensemble_pob.append(peak / base)
                # plt.plot(proc_arr[:,1])
                # plt.show()
            elif scatter_mode == 's':
                proc_arr = rpro.proc2s(self.pix_size, self.cam_length, self.wavelength)
                target = f'{reduction_path}{self.tag}_{k}_reduced_s'
                np.save(target, profile)
        np.save(f'{reduction_path}{self.tag}_peak.npy', np.array(ensemble_peak))
        np.save(f'{reduction_path}{self.tag}_base.npy', np.array(ensemble_base))
        np.save(f'{reduction_path}{self.tag}_pob.npy', np.array(ensemble_pob))
        plt.plot(ensemble_peak[:])
        plt.plot(ensemble_base[:])
        plt.plot(ensemble_pob[:])
        plt.show()

    def scatter_shot_inspect(self, dbin_folder='h5_frames', sample_size=10, dump=False):
        dbin_ls = sorted_nicely(glob.glob(f'{self.apath}{dbin_folder}/*.dbin'))
        for count in range(sample_size):
            rand_indx = random.randint(0, len(dbin_ls))
            print(rand_indx)
            print(f'<scatter_shot_inspect> frame: {dbin_ls[rand_indx]}')
            d = self.read_dbin(dbin_ls[rand_indx])
            frame = d * self.mask
            plt.figure()

            plt.imshow(frame)
            plt.clim(0, 50)
            np.savetxt(f'{self.apath}{self.tag}_sshot_{count}.txt', frame)
            plt.title(f'{dbin_ls[rand_indx]}')
            plt.figure()
            profile = self.frm_integration(frame)
            plt.plot(profile)
            plt.figure()
            frame_polar = warp_polar(np.transpose(frame), center=self.image_center, radius=1024)
            print(frame_polar.shape)
            plt.imshow(frame_polar)
            plt.figure()
            plt.plot(np.sum(frame_polar[:, 117:128], axis=1))
            plt.show()

    def define_average_profile(self, folder='1d_profiles', limit=10000):
        prf_list = glob.glob(f'{self.apath}{folder}/*.npy')
        parent_mf = f'/data/xfm/{self.experiment_id}/analysis/eiger/SAXS/{self.group}/{self.tag}/h5_frames/{self.tag}_manifest.txt'
        prf_list = sorted_nicely(prf_list)
        print(f'{prf_list[0]}')
        print(f'{prf_list[-1]}')
        prf_num = len(prf_list)
        print(f"<define_average_profile> number of profiles: {prf_num}")
        if limit != 10000:
            limit = prf_num
        measure = np.load(prf_list[0])
        print(f'<define_average_profile>{measure.shape}')
        tau_array = np.zeros((measure.shape[0], limit))
        print(f'<define_average_profile>{tau_array.shape}')
        for k, arr in enumerate(prf_list[:limit]):
            prf = np.load(prf_list[k])
            tau_array[:, k] = prf[:, 1]
        average_tau = np.average(tau_array, axis=1)
        print(f'<define_average_profile>{average_tau.shape}')
        self.average_profile = np.column_stack((prf[:, 0], average_tau[:]))
        plt.plot(self.average_profile[:, 0], self.average_profile[:, 1])
        plt.show()
        return self.average_profile

    def calc_rfactor(self, profile):
        delta = 0.0
        yobs = 0.0
        for i, dp in enumerate(profile):
            delta = delta + (profile[i, 1] - self.average_profile[i, 1]) ** 2
            yobs = yobs + (profile[i, 1]) ** 2
        r_p = np.sqrt(delta / yobs)
        # print(f'R_p : {r_p}')
        return r_p

    def define_parent_manifest(self, pmf_path):
        frm_list = []
        line_list = []
        if pmf_path == '':
            pmf_path = f'/data/xfm/{self.experiment_id}/analysis/eiger/SAXS/{self.group}/{self.tag}/h5_frames/{self.tag}_manifest.txt'
        with open(pmf_path, 'r') as f:
            lines = f.readlines()
            print(f'<define_parent_manifest> Total of {len(lines)} frames in parent manifest')
            for line in lines:
                line_list.append(line)
                sploot = line.split('/')
                splat = sploot[-1].split('.')[0]
                # print(splat)
                frm_list.append(splat)
        print(f'<define_parent_manifest> Parent manifest has {len(frm_list)} frames')
        return frm_list, line_list

    def grab_parent_prfs(self, frm_list, folder):
        prf_list = []
        for frm in frm_list:
            # print(f'frm {frm}')
            prf_list.append(f'{self.apath}{folder}/{frm}_reduced_q_m.npy')
        return prf_list

    def calc_subset_average(self, prf_list):
        limit = len(prf_list)
        measure = np.load(prf_list[0])
        print(measure.shape)
        tau_array = np.zeros((measure.shape[0], limit))
        print(tau_array.shape)
        for k, arr in enumerate(prf_list[:limit]):
            prf = np.load(prf_list[k])
            tau_array[:, k] = prf[:, 1]
        average_tau = np.average(tau_array, axis=1)
        print(average_tau.shape)
        subset_ap = np.column_stack((prf[:, 0], average_tau[:]))
        plt.figure()
        plt.title('subset average')
        plt.plot(subset_ap[:, 0], subset_ap[:, 1])
        plt.show()
        return subset_ap

    def calc_subset_rfactor(self, profile, average):
        delta = 0.0
        yobs = 0.0
        # print(profile.shape)
        # print(average.shape)
        for i, dp in enumerate(profile):
            delta = delta + (profile[i, 1] - average[i, 1]) ** 2
            yobs = yobs + (profile[i, 1]) ** 2
        r_p = np.sqrt(delta / yobs)
        # print(f'R_p : {r_p}')
        # plt.figure()
        # plt.plot(profile[:,0],profile[:,1])
        # plt.plot(average[:,0],average[:,1])
        # plt.show()
        return r_p

    def make_filtered_manifest(self, filtered_indices, frm_list, line_list, mf_path, even_mf_path, odd_mf_path):
        count = 0
        odd_count = 0
        even_count = 0
        with open(mf_path, 'w') as out:
            with open(even_mf_path, 'w') as out_even:
                with open(odd_mf_path, 'w') as out_odd:
                    for k, index in enumerate(filtered_indices):
                        int_ind = int(index)
                        out.write(line_list[int_ind])
                        count += 1
                        if k % 2 == 0:
                            out_even.write(line_list[int_ind])
                            even_count += 1
                        else:
                            out_odd.write(line_list[int_ind])
                            odd_count += 1
        print(f'<make_filtered_manifest> I wrote {count} files to the manifest {mf_path}')
        print(f'<make_filtered_manifest> I wrote {odd_count} files to the manifest {odd_mf_path}')
        print(f'<make_filtered_manifest> I wrote {even_count} files to the manifest {even_mf_path}')
        return line_list

    def filter_against_average(self, folder='1d_profiles', limit=10000, rfac_threshold=1.0, itera=0, parent_mf='',
                               inspect=False, qlims=[0, 10]):
        frm_list, line_list = self.define_parent_manifest(parent_mf)
        prf_list = self.grab_parent_prfs(frm_list, folder)
        filtered_indices = []
        mf_path = f'/data/xfm/{self.experiment_id}/analysis/eiger/SAXS/{self.group}/{self.tag}/h5_frames/{self.tag}_average_filter_manifest_{itera}.txt'
        odd_mf_path = f'/data/xfm/{self.experiment_id}/analysis/eiger/SAXS/{self.group}/{self.tag}/h5_frames/{self.tag}_average_filter_manifest_{itera}_odd.txt'
        even_mf_path = f'/data/xfm/{self.experiment_id}/analysis/eiger/SAXS/{self.group}/{self.tag}/h5_frames/{self.tag}_average_filter_manifest_{itera}_even.txt'
        prf_list = sorted_nicely(prf_list)
        print(prf_list[0])
        print(prf_list[-1])
        self.rfactor_array = []
        prf_num = len(prf_list)
        subset_ap = self.calc_subset_average(prf_list) + 1.0
        subset_ap = trim_to_qlims(qlims, subset_ap)
        print(f'len of trimmed subset_ap {subset_ap.shape}')
        # print(f'len(prf_list) {len(prf_list)}')
        print(f'len(self.rfactor_array) {len(self.rfactor_array)}')
        for k, arr in enumerate(prf_list[:limit]):
            prf = np.load(prf_list[k]) + 1.0
            prf = trim_to_qlims(qlims, prf)
            self.rfactor_array.append(self.calc_subset_rfactor(prf, subset_ap))
        plt.figure()
        plt.plot(self.rfactor_array[:])
        plt.show()
        plt.figure()
        plt.ylabel('Counts')
        plt.xlabel('R factor')
        plt.hist(self.rfactor_array, bins=20, range=(0.0, 0.1))
        plt.show()
        print(f'<filter_against_average> filtering with limit <= {rfac_threshold}')
        print(f'len(self.rfactor_array) {len(self.rfactor_array)}')
        for k, rfac in enumerate(self.rfactor_array):
            if rfac <= rfac_threshold:
                filtered_indices.append(k)
                if inspect:
                    plt.figure()
                    plt.title(f'{self.calc_subset_rfactor(prf, subset_ap)}')
                    plt.plot(prf[:, 0], prf[:, 1])
                    plt.plot(subset_ap[:, 0], subset_ap[:, 1])
                    plt.show()
        print(f'<filter_against_average> a total of {len(filtered_indices)} profles < {rfac_threshold}')
        self.make_filtered_manifest(filtered_indices, frm_list, line_list, mf_path, odd_mf_path, even_mf_path)

    def quick_mask(self, frame):
        mfrm = ma.masked_where(frame > self.max_lim, frame)
        return mfrm

    def quick_int_filter(self, threshold):
        print(len(self.frm_indexes))
        self.frm_indexes = np.delete(self.frm_indexes, np.where(self.tag_int_wt[:, 1] < threshold))
        print(len(self.frm_indexes))

    def mem_run_sum(self):
        sum_data = np.sum(self.run_data_array, 0)
        print(f'dsum.size = {sum_data.size}')
        self.run_data_sum = sum_data
        return sum_data

    def mem_run_avg(self):
        avg_data = np.average(self.run_data_array, axis=0)
        print(f'avg_data.size = {avg_data.size}')
        self.run_data_avg = avg_data
        return avg_data

    def show_overview(self):


    def quick_overview(self, run_limit=0, show=False):
        print(f'<fluxfm.overview> Analyzing run {self.tag}')
        self.tag_int_wt = []
        for k, h5 in enumerate(self.h5ls[:run_limit]):
            print('<fluxfm.overview> h5:', h5)
            print('<fluxfm.overview> Reading:', self.dpath + h5)
            with h5py.File(self.dpath + h5) as f:
                d = np.array(f['entry/data/data'])
                if k == 0:
                    self.run_data_array = d
                    self.mask = self.gen_mask(d[0], max_lim=self.max_lim)
                    print(f'<fluxfm.overview> {self.mask.shape}')
                else:
                    self.run_data_array = np.concatenate((self.run_data_array, d), axis=0)
                h5_sum = np.sum(d, axis=0)
                np.savetxt(f'{self.apath}{self.tag}_sum_h5_{k + 1}_red.dat',
                           self.frm_integration(h5_sum * self.mask, npt=2250))
                np.save(f'{self.apath}{self.tag}_sum_h5_{k + 1}_red.npy',
                        self.frm_integration(h5_sum * self.mask, npt=2250))
                print(f'<fluxfm.quick_overview> {h5} shape:  {d.shape}')
                print(f'<fluxfm.quick_overview> {self.run_data_array.shape}')
        self.run_data_sum = self.mem_run_sum()
        self.run_data_avg = self.mem_run_avg()
        print(f'<fluxfm.quick_overview> Writing overview sum to:{self.scratch}{self.tag}_sum_reduced_q.npy')
        np.save(self.scratch + self.tag + '_sum.npy', self.run_data_sum)
        np.save(f'{self.apath}{self.tag}_sum_red.npy', self.frm_integration(self.run_data_sum * self.mask, npt=2250))
        print(f'<fluxfm.quick_overview> Writing overview mean to:{self.scratch}{self.tag}_avg_reduced_q.npy')
        np.save(self.scratch + self.tag + '_avg.npy', self.run_data_avg)
        np.save(f'{self.apath}{self.tag}_avg_red.npy', self.frm_integration(self.run_data_avg * self.mask, npt=2250))
        print(f'median dsum {np.median(self.run_data_sum)}')
        print(f'media avg {np.median(self.run_data_avg)}')
        if show:
            plt.figure(f'{self.tag} Masked sum')
            plt.imshow(self.run_data_sum * self.mask)
            plt.clim(0, np.median(self.run_data_sum) * 3)
            plt.figure(f'{self.tag} Masked average')
            plt.imshow(self.run_data_avg * self.mask)
            plt.clim(0.0, np.median(self.run_data_avg) * 3)
            plt.show()

    def run_movie(self, run_id=0, step=100, cmin=0, cmax=1000):
        print(self.h5ls[0])
        for h5 in self.h5ls[run_id:run_id + 1]:
            print(h5)
            print('<overview> h5:', h5)
            print('<overview> Reading:', self.dpath + h5)
            with h5py.File(self.dpath + h5) as f:
                d = np.array(f['entry/data/data'])
                self.run_data_array = d

            frames = []
            fig = plt.figure()
            for k, img in enumerate(self.run_data_array[::step]):
                mfrm = self.quick_mask(img)
                print(k)
                frames.append([plt.imshow(img * mfrm, animated=True, clim=[cmin, cmax])])

            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
            ani.save(f'{self.apath}{self.tag}_{run_id}.mp4')
            print(f'{self.apath}{self.tag}_{run_id}.mp4')
            plt.show()
