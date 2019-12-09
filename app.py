# -*- coding: utf-8 -*-
"""
# Calculation of G' and G'' from QCM-D data

This script controls the GUI functions for calculating
viscoelastic properties from QCM-D data by
fitting $\Delta$F and $\Delta$D QCM data to the
Kelvin-Voigt model to obtain viscosity and shear modulus of
an adlayer film for calculation of  G' and G'' (elastic and loss moduli).

Use the *User Inputs* section to input experimental data. Run the
viscoelastic model. Once the solution is found, you may adjust
the range of μ and η values to limit the search space and obtain
a solution with higher precision.  

For more information about the viscoelastic model, see:
1.   Voinova, M.V., Rodahl, M., Jonson, M. and Kasemo, B., 1999. Viscoelastic
acoustic response of layered polymer films at fluid-solid interfaces:
continuum mechanics approach. Physica Scripta, 59(5), p.391.
https://iopscience.iop.org/article/10.1238/Physica.Regular.059a00391/meta
2.   Liu, S.X. and Kim, J.T., 2009. Application of Kelvin—Voigt model in
quantifying whey protein adsorption on polyethersulfone using QCM-D.
JALA: Journal of the Association for Laboratory Automation, 14(4),
pp.213-220.
https://journals.sagepub.com/doi/full/10.1016/j.jala.2009.01.003

Created on Dec 9 2019
@author: ericmuckley@gmail.com

"""


# core GUI libraries
from PyQt5 import QtWidgets, uic, QtCore#, QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog
#from PyQtCore import QRunnable, QThreadPool, pyqtSlot

# from threading import Thread

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# change matplotlib settings to make plots look nicer
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3


class Worker(QtCore.QRunnable):
    """Class to start a new worker thread for background tasks.

    Call this thread inside a main GUI function by:
    worker = Worker(self.function_to_execute)  # pass other args here
    self.threadpool.start(worker)
    where self.function_to_execute is the function to run and its args
    """
    def __init__(self, fn, *args, **kwargs):
        """This allows the Worker class to take any function as an
        argument, along with args, and run it in a separate thread."""
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self):
        """Take a function and its args which were passed to the Worker
        class and execute it in a new thread."""
        self.fn(*self.args, **self.kwargs)
  


class App(QMainWindow):
    """Class which creates the main window of the application."""

    # load Qt designer XML .ui GUI file
    Ui_MainWindow, QtBaseClass = uic.loadUiType('ui.ui')

    def __init__(self):

        super(App, self).__init__()
        self.ui = App.Ui_MainWindow()
        self.ui.setupUi(self)

        # initialize multithreading
        self.threadpool = QtCore.QThreadPool()




        # self.move(150, 150)  # set initial position of the window

        # create timer which updates fields on GUI (set interval in ms)
        # self.timer = QtCore.QTimer(self)
        # self.timer.timeout.connect(self.main_loop)
        # self.timer.start(int(self.ui.set_main_loop_delay.value()))

        # assign functions to top menu items
        # example: self.ui.menu_item_name.triggered.connect(self.function_name)
        #self.ui.actionShowfiledir.triggered.connect(self.show_directory)
        #self.ui.actionChangefiledir.triggered.connect(self.set_directory)
        self.ui.actionQuit.triggered.connect(self.quitapp)

        # assign actions to GUI buttons
        # example: self.ui.BUTTON_NAME.clicked.connect(self.FUNCTION_NAME)
        self.ui.run_model.clicked.connect(self.run_model)
        #self.ui.setdirectory.clicked.connect(self.setdirectory)
        #self.ui.addsample.clicked.connect(self.add_sample)
        #self.ui.createplot.clicked.connect(self.create_plot)
        #self.ui.save_file.clicked.connect(self.save_file)
        
        
        #self.ui.test_pause_button.clicked.connect(self.pause)
        
        
        #self.ui.importdata.clicked.connect(self.import_dict)
        
        
        # assign actions to checkboxes
        # example: self.ui.CHECKBOX.stateChanged.connect(self.FUNCTION_NAME)

        #self.ui.savesampleinfo.setEnabled(False)

        # initialize some settings
        # self.eis_busy = False

        # initialize file-saving variables
        # self.df_i = 0
        # self.save_file_dir = None
        # self.start_time = time.time()
        # self.start_date = time.strftime('%Y-%m-%d_%H-%M_')
        # this opens file diaglog for saving
        # self.set_file_save_directory()
        
        
        # set default data folder and create it if it doesn't exist
        self.filedir = os.getcwd()+'\\data'
        if not os.path.exists(self.filedir):
            os.makedirs(self.filedir)
       
        

        
        # initialize plots so we can check later if they exist
        self.p1 = None

        self.random_list = []


# %% ----------- system control functions ------------------------------

    # file I/O utilities ---------------------------------------------------

    def save_file(self):
        """Save some data to file."""
        filename = 'test_file_name'
        filepath = self.filedir + '\\' + filename + '.csv'
        df = pd.DataFrame(columns=['a', 'b', 'c'],
                          data=np.random.random((100, 3)))
        df.to_csv(filepath)
        self.ui.outbox.append('\nFile saved to ' + filepath)

    def set_directory(self):
        """Set the directory for saving files."""
        self.filedir = str(QFileDialog.getExistingDirectory(
                self, 'Select a directory for storing data'))
        self.ui.outbox.append('\nFile directory is set to ' + self.filedir)


    def show_directory(self):
        """Show the file directory in the output box."""
        self.ui.outbox.append('\nFile directory is set to ' + self.filedir)





    def execute_during_pause(self):
        print("Thread start") 
        
        for i in range(6):
            a = np.random.random()
            print(a)
            print(i)
            print(time.ctime())
            self.ui.outbox.append(str(i))
            time.sleep(1)
            self.random_list.append(a)
        print("Thread complete")
        self.ui.outbox.append(str(self.random_list))

    def pause(self):
        """Pause the gui to check responsivity."""
        worker = Worker(self.execute_during_pause)  # pass other args here
        self.threadpool.start(worker)











    def get_ui_inputs(self):
        """Get a dictionary of inputs from the UI."""
        uidict = {
                'f0': float(self.ui.f0.currentText())*1e6,
                'n': int(self.ui.n.currentText()),
                'rho': float(self.ui.film_density.value()),
                'h': float(self.ui.film_thickness.value())*1e-9,
                'medium': str(self.ui.medium.currentText()),
                'df_exp': float(self.ui.df_exp.value()),
                'dd_exp': float(self.ui.dd_exp.value()),
                'mu_low': int(self.ui.mu_exp_low.value()),
                'mu_high': int(self.ui.mu_exp_high.value()),
                'eta_low': int(self.ui.eta_exp_low.value()),
                'eta_high': int(self.ui.eta_exp_high.value())}
        return uidict
                

    def run_model(self):
        """Run modeling of the QCM-D dtaa using inputs from UI. """
        # get dictionary of input values from UI
        uidict = self.get_ui_inputs()
        results = uidict.copy()
        
        for key in uidict:
            print(key)
            print(str(uidict[key]))


        # get mesh of mu and eta values
        mu_mesh, eta_mesh = self.get_mu_eta_mesh(uidict)
        
        # calculate theoretical DF and DD values across the mu and eta grid
        df_surf, dd_surf = self.kelvin_voigt(10**mu_mesh,
                                             10**eta_mesh,
                                             rho_f=uidict['rho'],
                                             h_f=uidict['h'],
                                             medium=uidict['medium'],
                                             f0=uidict['f0'],
                                             n=uidict['n'])
        
        # plot delta F heatmap
        df_cont_plot = plt.contour(mu_mesh, eta_mesh,
                                   df_surf, uidict['df_exp'])
        if self.ui.contour_plots.isChecked():
            plt.contourf(mu_mesh, eta_mesh, df_surf, 50, cmap='rainbow')
            self.plot_setup(title='Delta F (Hz/cm^2)',
                       labels=['Log (mu)', 'Log (eta)'], colorbar=True)
            plt.show()
        
        # plot delta D heatmap
        dd_cont_plot = plt.contour(mu_mesh, eta_mesh,
                                   dd_surf, uidict['dd_exp'])
        if self.ui.contour_plots.isChecked():
            plt.contourf(mu_mesh, eta_mesh, dd_surf, 50, cmap='rainbow')
            self.plot_setup(title='Delta D (x10^-6)',
                       labels=['Log (mu)', 'Log (eta)'], colorbar=True)
            plt.show()
        
        # extract contours which correspond to experimental values
        df_cont = self.get_contour(df_cont_plot)
        dd_cont = self.get_contour(dd_cont_plot)
        # find intersection of solutions
        intersection_list = np.array(
                self.find_intersections(df_cont, dd_cont))
        
        # check if there are any solutions
        if len(intersection_list) > 0:
        
            # select the 0th-order solution
            sol = sorted(intersection_list, key = lambda i: float(i[1]))[-1]
            plt.scatter(sol[0], sol[1], marker='x', s=300, c='k', label='solution')
            #plt.scatter(intersection_list[:,0], intersection_list[:,1])
        
            # plot contour intersection
            plt.scatter(df_cont[:, 0], df_cont[:, 1], s=1, c='b', label='df')
            plt.scatter(dd_cont[:, 0], dd_cont[:, 1], s=1, c='r', label='dd')
            self.plot_setup(title='Contour intersection',
                            labels=['Log (mu)', 'Log (eta)'], legend=True)
            plt.show()
        
            # get calculated mu and eta values, along with G' and G'' 
            mu, eta, = 10**sol[0], 10**sol[1]
            Gp, Gdp = mu, 2*np.pi*uidict['f0']*eta
            # get fitted df and dd values
            df_fit, dd_fit = self.kelvin_voigt(mu, eta,
                                               rho_f=uidict['rho'],
                                               h_f=uidict['h'],
                                               medium=uidict['medium'],
                                               f0=uidict['f0'],
                                               n=uidict['n'])
            pen_dep = self.get_penetration_depth(uidict['f0'],
                                                 eta,
                                                 uidict['rho'])
        
        
            results['df_fit'] = df_fit
            results['dd_fit'] = dd_fit
            results['mu'] = mu
            results['eta'] = eta
            results['penetration_depth']: pen_dep
            results["G'"] = Gp
            results["G''"] = Gdp
        
            print('Found %i solutions. First-order solution:' %len(intersection_list))
            [print('%s: %0.4e' %(key, results[key])) for key in results]
        
        else:
            self.ui.outbox.append(
                    '\n\nNo solutions exist with these parameters.')

























    def get_mu_eta_mesh(self, uidict, step_num=100):
        """Create mesh of mu and eta valuses using inputs on UI."""
        # get 2D mesh grid points of log mu and eta values
        mu_mesh, eta_mesh = np.meshgrid(
                np.linspace(uidict['mu_low'], uidict['mu_high'],
                            step_num).astype(float),
                np.linspace(uidict['eta_low'], uidict['eta_high'],
                            step_num).astype(float))
        return mu_mesh, eta_mesh
        

    def plot_setup(labels=['X', 'Y'], fsize=20, setlimits=False,
                   title=None, legend=False, colorbar=False,
                   limits=[0,1,0,1], save=False, filename='plot.jpg'):
        """Creates a custom plot configuration to make graphs look nice.
        This can be called with matplotlib for setting axes labels,
        titles, axes ranges, and the font size of plot labels.
        This should be called between plt.plot() and plt.show() commands."""
        plt.xlabel(str(labels[0]), fontsize=fsize)
        plt.ylabel(str(labels[1]), fontsize=fsize)
        fig = plt.gcf()
        fig.set_size_inches(6, 4)
        if title:
            plt.title(title, fontsize=fsize)
        if legend:
            plt.legend(fontsize=fsize-4)
        if setlimits:
            plt.xlim((limits[0], limits[1]))
            plt.ylim((limits[2], limits[3]))
        if colorbar:
            plt.colorbar()
        if save:
            fig.savefig(filename, dpi=120, bbox_inches='tight')
            plt.tight_layout()
    
    def kelvin_voigt(mu_f, eta_f, rho_f=1e3, h_f=1e-6, n=1, f0=5e6,
                    medium='air'):
        """ 
        The Kelvin-Voigt model comes from eqns (15) in the paper by 
        Voinova: Vionova, M.V., Rodahl, M., Jonson, M. and Kasemo, B., 1999.
        Viscoelastic acoustic response of layered polymer films at fluid-solid
        interfaces: continuum mechanics approach. Physica Scripta, 59(5), p.391.
        Reference: https://github.com/88tpm/QCMD/blob/master
        /Mass-specific%20activity/Internal%20functions/voigt_rel.m.
        
        This function solves for Delta f and Delta d of thin adlayer on QCM.
        It differs from voigt because it calculates relative to an
        unloaded resonator.
        Inputs
            mu_f = shear modulus of film in Pa
            eta_f = shear viscosity of film in Pa s
            rho_f = density of film in kg m-3
            h_f = thickness of film in m
            n = crystal harmonic number
            f0 = fundamental resonant frequency of crystal in Hz      
        Output
            deltaf = frequency change of resonator
            deltad =  dissipation change of resonator
        """
        # define properties of QCM crystal
        w = 2*np.pi*f0*n  # angular frequency
        mu_q = 2.947e10  # shear modulus of AT-cut quatz in Pa
        rho_q = 2648  # density of quartz (kg/m^3)
        h_q = np.sqrt(mu_q/rho_q)/(2*f0)  # thickness of quartz
        # define properties of medium
        if medium == 'air':
            rho_b = 1.1839  # density of bulk air (25 C) in kg/m^3
            eta_b = 18.6e-6  # viscosity of bulk air (25 C) in Pa s
        if medium == 'liquid':
            rho_b = 1000  # density of bulk water in kg/m^3
            eta_b = 8.9e-4  # viscosity of bulk water in Pa s
        # define equations from the Kelvin-Voigt model in publication
        # eqn 14
        kappa_f = eta_f-(1j*mu_f/w)
        # eqn 13
        x_f = np.sqrt(-rho_f*np.square(w)/(mu_f + 1j*w*eta_f))
        x_b = np.sqrt(1j*rho_b*w/eta_b)
        # eqn 11 after simplification with h1 = h2 and h3 = infinity
        A = (kappa_f*x_f+eta_b*x_b)/(kappa_f*x_f-eta_b*x_b)
        # eqn 16
        beta = kappa_f*x_f*(1-A*np.exp(2*x_f*h_f))/(1+A*np.exp(2*x_f*h_f))
        beta0 = kappa_f*x_f*(1-A)/(1+A)
        # eqn 15
        df = np.imag((beta-beta0)/(2*np.pi*rho_q*h_q))
        dd = -np.real((beta-beta0)/(np.pi*f0*n*rho_q*h_q))*1e6
        return df, dd
    
    def get_contour(cont_plot):
        """Get ordered pairs of contour lines from a contour plot.
        Input should be defined as:
        cont_plot = plt.contour(x_mesh, y_mesh, z_surf, contour_value)"""
        # extract contour paths from plot
        paths = [path.vertices for path in cont_plot.collections[0].get_paths()]
        if paths:
            # stack all contour paths in a single 2D array
            return np.vstack(paths)
        else:
            return []
    
    def find_intersections(op_list1, op_list2):
        """Find all intersections between two curves. Curves are defined by lists
        of ordered pairs (x, y).
        Returns an empty list if no intersections are found."""
        intersections = []
        # check if both curves contain more than 1 point:
        if len(op_list1) > 1 and len(op_list2) > 1:
            # loop over each pair of line segments
            for i1 in range(len(op_list1)-1):
                for i2 in range(len(op_list2)-1):
                    # create segment from the first set of points
                    seg1 = LineString([(op_list1[i1][0], op_list1[i1][1]),
                                    (op_list1[i1+1][0], op_list1[i1+1][1])])
                    # create segment from the second set of points
                    seg2 = LineString([(op_list2[i2][0], op_list2[i2][1]),
                                    (op_list2[i2+1][0], op_list2[i2+1][1])])
                    # check if segment from set-1 intersects segment from set-2
                    if seg1.intersects(seg2):
                        avg_x = np.mean([op_list1[i1][0], op_list1[i1+1][0],
                                        op_list2[i2][0], op_list2[i2+1][0]])
                        avg_y = np.mean([op_list1[i1][1], op_list1[i1+1][1],
                                        op_list2[i2][1], op_list2[i2+1][1]])
                        intersections.append([avg_x, avg_y])
        return intersections
    
    def get_penetration_depth(freq, eta, rho):
        """Calculate penetration depth of acoustic wave using the QCM
        resonant requency (freq), adlayer visacosity (eta), and adlayer
        density (rho)."""
        return np.sqrt(eta / (np.pi * freq * rho))





    # ---------------------------------------------------------------------
    
  

    def main_loop(self):
        """Execute a main loop repeatedly while the app is running."""
        pass

    def quitapp(self):
        """Quit the application."""
        self.deleteLater()
        # self.timer.stop()  # stop timer
        # close app window
        self.close()  
        # kill python kernel
        sys.exit()  





# %% -------------------------- run application ----------------------------


if __name__ == "__main__":
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    window = App()
    window.show()
    sys.exit(app.exec_())
