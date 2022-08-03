import h5py
import numpy as np
import time
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
import configparser
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
    snipped_int = []
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


class XfmHfiveDataset:

    def __init__(self, configpath=''):

        self.configpath = configpath

        # Parameters passed from config file:
        self.nx = 1062
        self.ny = 1028
        self.max_px_count = 4.29E9
        self.cam_length = 1.0
        self.wavelength = 1.0
        self.pix_size = 1.0
        self.experiment_id = ''
        self.maia_num = 0

        # Set in runner
        self.group = ''
        self.tag = ''
        self.dpath = ''
        self.apath = ''
        self.scratch = ''
        self.mfpath = ''
        self.image_center = (0, 0)

        # Internals
        self.h5ls = []

        self.mask = []
        self.bboxes = []
        self.circs = []

        self.run_data_array = []
        self.run_data_sum = []
        self.red_data_sum = None
        self.run_data_avg = []
        self.red_data_avg = None

        self.average_profile = []
        self.rfactor_array = []
        self.tag_int_wt = []
        self.frm_indexes = []

        # Run initialization
        self.parameter_pass()

    def parameter_pass(self):
        print(f'<fluxfm.parameter_pass> Initializing analysis module...')
        conpar = configparser.RawConfigParser()
        conpar.read(self.configpath)
        param_dict = dict(conpar.items('FLUXFM_CONFIG'))
        self.nx = int(param_dict['eiger_nx'])
        self.ny = int(param_dict['eiger_ny'])
        self.max_px_count = float(param_dict['max_px_count'])
        self.cam_length = float(param_dict['cam_length'])
        self.wavelength = float(param_dict['wavelength'])
        self.pix_size = float(param_dict['pix_size'])
        self.experiment_id = str(param_dict['experiment_id'])
        self.maia_num = int(param_dict['maia_num'])
        self.dpath = str(param_dict['experiment_data_path'])
        self.apath = str(param_dict['experiment_analysis_path'])

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

    def gen_mask(self, image, max_lim, bboxes=False, circs=False, dump=True):
        print('<fluxfm.gen_mask> Generating mask...')
        self.mask = np.ones(image.shape)
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

    def atomize_reduce_h5(self, img_folder='h5_frames', profile_folder='1d_profiles', masked=False):
        """
        Each h5 in self.h5ls is separated into separated 2d images stored as npy arrays.
        These images are also reduced (masked by default) for filtering
        :param masked: boolean, if True, mask the 2d image, reduced data is always masked
        :param img_folder: location where 2d npy arrays are stored
        :param profile_folder: location where 1d reduced arrays are stored
        :return:
        """
        self.mask = np.load(f'{self.apath}{self.tag}_mask.npy')
        atom_path = f'{self.apath}{img_folder}/'
        red_path = f'{self.apath}{profile_folder}/'
        if not os.path.exists(atom_path):
            os.makedirs(atom_path)
        if not os.path.exists(red_path):
            os.makedirs(red_path)
        print(f'<atomize_reduce_h5> Atomizing to {atom_path}')
        print(f'<atomize_reduce_h5> Reducing to {red_path}')
        start = time.time()
        with open(f'{atom_path}{self.tag}_manifest.txt', 'w') as f:
            for k, h5 in enumerate(sorted_nicely(self.h5ls)):
                print(f'<atomize_reduce_h5> Atomizing {h5}...')
                with h5py.File(self.dpath + h5) as h:
                    d = np.array(h['entry/data/data'])
                    for shot in range(d.shape[0]):
                        if shot % 100 == 0:
                            print(f'<atomize_reduce_h5> {shot}/{d.shape[0]} frames generated')
                        if masked:
                            frame = d[shot, :, :] * self.mask
                        else:
                            frame = d[shot, :, :]
                        profile = self.frm_integration(frame * self.mask)
                        np.save(f'{red_path}{self.tag}_{k}_{shot}_red.npy', profile)
                        np.save(f'{atom_path}{self.tag}_{k}_{shot}.npy', frame)
                        f.write(f'{atom_path}{self.tag}_{k}_{shot}.npy\n')
        print(f'<atomize_reduce_h5> ...complete in {time.time() - start} seconds')
        print(f'<atomize_reduce_h5> File manifest written to {atom_path}{self.tag}_manifest.txt')

    def frm_integration(self, frame, unit="q_nm^-1", npt=2250):
        """
        Perform azimuthal integration of frame array
        :param frame: numpy array containing 2D intensity
        :param unit:
        :param npt:
        :return: two-col array of q & intensity.
        """
        ai = AzimuthalIntegrator()
        ai.setFit2D(directDist=self.cam_length,
                    centerX=self.image_center[0],
                    centerY=self.image_center[1],
                    pixelX=self.pix_size,
                    pixelY=self.pix_size)
        ai.wavelength = self.wavelength / 1e10
        integrated_profile = ai.integrate1d(data=frame, npt=npt, unit=unit)
        return np.transpose(np.array(integrated_profile))

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
            prf_list.append(f'{self.apath}{folder}/{frm}_red.npy')
        return prf_list

    def calc_subset_average(self, prf_list):
        limit = len(prf_list)
        measure = np.load(prf_list[0])
        print(f'<fluxfm.calc_subset_average> profile array shape: {measure.shape}')
        tau_array = np.zeros((measure.shape[0], limit))
        print(f'<fluxfm.calc_subset_average> average array shape: {tau_array.shape}')
        for k, arr in enumerate(prf_list[:limit]):
            prf = np.load(prf_list[k])
            tau_array[:, k] = prf[:, 1]
        average_tau = np.average(tau_array, axis=1)
        subset_ap = np.column_stack((prf[:, 0], average_tau[:]))
        plt.figure()
        plt.title('subset average')
        plt.plot(subset_ap[:, 0], subset_ap[:, 1])
        plt.xlabel('q')
        plt.ylabel('intensity')
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

    def make_filtered_manifest(self, filtered_indices, line_list, mf_path, even_mf_path, odd_mf_path):
        """
        Write out filtered manifest files for py3padf using filtered indices. This function can be
        expanded if some other form of filtering is used
        :param filtered_indices: List of file indices that pass through some filter
        :param line_list: list of paths
        :param mf_path: string, path to manifest file
        :param even_mf_path: string, path to even manifest file
        :param odd_mf_path: string, path to odd manifest file
        :return: passes line list back if required
        """
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
                               inspect=False, qlims=(1e7, 1e8)):
        # First grab the list of 1D profiles
        frm_list, line_list = self.define_parent_manifest(parent_mf)
        prf_list = self.grab_parent_prfs(frm_list, folder)
        filtered_indices = []
        # Paths to the resulting manifest files. Split into odd/even pairs for easy convergence testing
        mf_path = f'/data/xfm/{self.experiment_id}/analysis/eiger/SAXS/{self.group}/{self.tag}/h5_frames/{self.tag}_average_filter_manifest_{itera}.txt'
        odd_mf_path = f'/data/xfm/{self.experiment_id}/analysis/eiger/SAXS/{self.group}/{self.tag}/h5_frames/{self.tag}_average_filter_manifest_{itera}_odd.txt'
        even_mf_path = f'/data/xfm/{self.experiment_id}/analysis/eiger/SAXS/{self.group}/{self.tag}/h5_frames/{self.tag}_average_filter_manifest_{itera}_even.txt'
        prf_list = sorted_nicely(prf_list)
        print(prf_list[0])
        print(prf_list[-1])
        self.rfactor_array = []
        # Generate the average of the files in the prf_list.
        # Note this can also be a different file to the whole average using the itera variable
        subset_ap = self.calc_subset_average(prf_list)
        # Trim the profile to the qlims
        subset_ap = trim_to_qlims(qlims, subset_ap)
        print(f'<fluxfm.filter_against_average> {subset_ap.shape}')
        if subset_ap.shape[0] == 0:
            print(f'WARNING:: qlims range outside integrated pattern range. Check qlims are in m (typical range 1e7 : '
                  f'1e8)')
        print(f'<fluxfm.filter_against_average>{len(self.rfactor_array)}')
        # For each profile, load, trim to the same qrange and calculate the R-factor
        for k, arr in enumerate(prf_list[:limit]):
            prf = np.load(prf_list[k]) + 1.0
            prf = trim_to_qlims(qlims, prf)
            self.rfactor_array.append(self.calc_subset_rfactor(prf, subset_ap))
        """
        Set up the figure here
        """
        plt.figure()
        plt.plot(self.rfactor_array[:])
        plt.show()
        plt.figure()
        plt.ylabel('Counts')
        plt.xlabel('R factor')
        plt.hist(self.rfactor_array, bins=20, range=(0.0, 0.1))
        plt.show()
        print(f'<fluxfm.filter_against_average> filtering with limit <= {rfac_threshold}')
        print(f'<fluxfm.filter_against_average> filtering total of {len(self.rfactor_array)} profiles')
        # Now filter against the R-factor limit
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
        mfrm = ma.masked_where(frame > self.max_px_count, frame)
        return mfrm

    def quick_int_filter(self, threshold):
        print(len(self.frm_indexes))
        self.frm_indexes = np.delete(self.frm_indexes, np.where(self.tag_int_wt[:, 1] < threshold))
        print(len(self.frm_indexes))

    def show_overview_figures(self):
        self.mask = np.load(f'{self.apath}{self.tag}_mask.npy')
        self.run_data_avg = np.load(f'{self.apath}{self.tag}_avg.npy')
        self.run_data_sum = np.load(f'{self.apath}{self.tag}_sum.npy')
        self.red_data_sum = np.load(f'{self.apath}{self.tag}_sum_red.npy')
        self.red_data_avg = np.load(f'{self.apath}{self.tag}_avg_red.npy')
        print(f'Plotting overview for {self.tag}')
        # Images
        plt.figure(f'{self.tag} Masked sum')
        plt.title(f'{self.tag} Masked sum')
        plt.imshow(self.run_data_sum * self.mask)
        plt.clim(0, np.median(self.run_data_sum) * 3)
        plt.figure(f'{self.tag} Masked average')
        plt.title(f'{self.tag} Masked average')
        plt.imshow(self.run_data_avg * self.mask)
        plt.clim(0.0, np.median(self.run_data_avg) * 3)
        # Reduced data
        plt.figure(f'{self.tag} Reduced masked sum')
        plt.title(f'{self.tag} Reduced masked sum')
        plt.xlabel('q / nm^{-1}')
        plt.ylabel('Intensity / arb. units')
        plt.plot(self.red_data_sum[:, 0], self.red_data_sum[:, 1])
        plt.figure(f'{self.tag} Reduced masked avg')
        plt.title(f'{self.tag} Reduced masked avg')
        plt.xlabel('q / nm^{-1}')
        plt.ylabel('Intensity / arb. units')
        plt.plot(self.red_data_avg[:, 0], self.red_data_avg[:, 1])
        plt.show()

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
                    self.mask = self.gen_mask(d[0], max_lim=self.max_px_count)
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
        self.run_data_sum = np.sum(self.run_data_array, 0)
        self.run_data_avg = np.average(self.run_data_array, axis=0)
        diff_data, diff_data_avg = self.calculate_difference_array(self.run_data_array)
        print(f'<fluxfm.quick_overview> Writing overview sum to:{self.scratch}{self.tag}_sum_reduced_q.npy')
        np.save(self.scratch + self.tag + '_sum.npy', self.run_data_sum)
        np.save(f'{self.apath}{self.tag}_sum_red.npy', self.frm_integration(self.run_data_sum * self.mask, npt=2250))
        print(f'<fluxfm.quick_overview> Writing overview mean to:{self.scratch}{self.tag}_avg_reduced_q.npy')
        np.save(self.scratch + self.tag + '_avg.npy', self.run_data_avg)
        np.save(f'{self.apath}{self.tag}_avg_red.npy', self.frm_integration(self.run_data_avg * self.mask, npt=2250))
        np.save(f'{self.apath}{self.tag}_avg_red_diff.npy',
                self.frm_integration(np.abs(diff_data_avg) * self.mask, npt=2250))
        if show:
            plt.figure(f'{self.tag} Masked sum')
            plt.imshow(self.run_data_sum * self.mask)
            plt.clim(0, np.median(self.run_data_sum) * 3)
            plt.figure(f'{self.tag} Masked average')
            plt.imshow(self.run_data_avg * self.mask)
            plt.clim(0.0, np.median(self.run_data_avg) * 3)
            plt.show()

    def calculate_difference_array(self, run_data_array):
        output_array = run_data_array * 0.0
        for i in np.arange(run_data_array.shape[0]):
            m = int(np.random.rand() * run_data_array.shape[0])
            if m>= run_data_array.shape[0]: m = run_data_array.shape[0] - 1
            output_array[i, :, :] = run_data_array[i, :, :] - run_data_array[m, :, :]

        return output_array, np.average(output_array, axis=0)


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
