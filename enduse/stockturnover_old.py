import numpy as np
import pandas as pd

class StockTurnoverModel:
    
    def __init__(self, model_config, calc_config):
        
        # initialize config vars
        self.model_config = model_config
        self.equipment_calc_config = calc_config['equipment_calc_config']
        self.calibration_config = calc_config['calibration_config']

        # initialize class attributes
        # will be set using dispatch functions
        
        # model configs
        self.building_stock = None
        self.saturation = None
        self.fuel_share = None
        self.efficiency_share = None
        self.equipment_measures = None
        self.equipment_standards = None
        self.join_order = None

        # equipment calc config
        self.equipment_label = None
        self.equipment_stock_label = None
        self.equipment_consumption_label = None
        self.equipment_measure_consumption_label = None
        self.equipment_sort_index = None
        self.equipment_calc_variables = None
        self.equipment_calc_index = None
        self.effective_useful_life_label = None
        self.efficiency_level_label = None
        self.ramp_efficiency_level_label = None
        self.ramp_efficiency_probability_label = None

        # calibration config
        self.customer_class_label = None
        self.time_label = None
        self.calibration_label = None
        self.goal_label = None
        self.adjusted_measure_consumption_label = None
        self.adjusted_consumption_label = None

        # populate class attributes
        self.dispatch_parser(self.model_config)
        self.build_join_order(self.model_config)
        self.build_from_config(self.equipment_calc_config)
        self.build_from_config(self.calibration_config)
        
    def dispatch_parser(self, model_config):
        # will create class attributes from config
        # order matters for join operations
        file_parser = {
            'building_stock': self.parse_building_stock, 
            'saturation': self.parse_common_files,
            'fuel_share': self.parse_common_files,
            'efficiency_share': self.parse_common_files,
            'equipment_measures': self.parse_common_files,
            'equipment_standards': self.parse_equipment_standards,
        }

        for i, x in file_parser.items():
            setattr(self, i, x(model_config[i]))

    def build_join_order(self, model_config):
        join_order = sorted(model_config, key=lambda x: model_config[x]['join_order'])
        setattr(self, 'join_order', tuple(join_order))

    def build_from_config(self, config):
        for i, x in config.items():
            setattr(self, i, x)
    
    def parse_building_stock(self, model_config):
        # TODO possible change output logic of building stock file to avoid this step
        building_stock = pd.read_csv(model_config['path']).set_index(list(model_config['index']))
        
        # need to do a cumulative sum of new building stock
        existing = building_stock.loc(axis=0)[:, :, 'Existing']
        new = building_stock.loc(axis=0)[:, :, 'New'].copy()
        new['Building Stock'] = new.groupby(level=new.index.names)['Building Stock'].cumsum()
        
        return pd.concat([existing, new]).reset_index()

    def parse_equipment_standards(self, model_config):
        '''Adjustment to allign equipment standards with base year and get'''
        standards = pd.read_csv(model_config['path'])
        
        # need to adjust start year of standards - Cadmus starts in first forecast year
        standards[model_config['start_time']] = np.where(
            standards[model_config['start_time']] == standards[model_config['start_time']].min(), 
            standards[model_config['start_time']].min() - 1, 
            standards[model_config['start_time']]
        )

        # need to map efficiency description to efficiency level
        measure_map = (
            self.equipment_measures.reset_index()[list(model_config['map_index'])]
            .drop_duplicates()
        )

        # get efficiency level for each equipment standards
        standards[model_config['map_relabel']] = (
            standards.merge(measure_map, left_on = list(model_config['map_left_on']), 
            right_on = list(model_config['map_right_on']))[model_config['map_label']]
        )

        return standards
    
    def parse_common_files(self, model_config):
        return pd.read_csv(model_config['path']).set_index(list(model_config['index'])).sort_index().reset_index()

    def run_model(self):
        model_data = self.create_model_data()
        model_data_t = self.interpolate_inputs(model_data)
        model_data_comb = self.join_data(model_data_t)
        return self.equipment_calc(model_data_comb)

    def join_data(self, model_data_t):
        '''Merge each file based on join order'''
        for n, (i, x) in enumerate(zip(self.join_order[:-1], self.join_order[1:])):
                idx = self.model_config[i]['join_index']
                if n == 0:
                    model_data = pd.merge(model_data_t[i], model_data_t[x], how='left', on=idx)
                else:
                    model_data = model_data.merge(model_data_t[x], how='left', on=idx)

        return model_data
    
    def create_model_data(self):
        
        # need to copy input data to avoid overwriting class attributes
        model_data = {
            'building_stock': self.building_stock.copy(),
            'saturation': self.saturation.copy(), 
            'fuel_share': self.fuel_share.copy(), 
            'efficiency_share': self.efficiency_share.copy(), 
            'equipment_measures': self.equipment_measures.copy(), 
            'equipment_standards': self.equipment_standards.copy()
        }
               
        return model_data

    def interpolate_inputs(self, model_data):
        
        model_data_t = {}

        for i, x in model_data.items():
            if i not in ['building_stock']:
                model_data_t[i] = self.tranpose_inputs(x, self.model_config[i])
            elif i in ['building_stock']:
                model_data_t[i] = x

        return model_data_t
    
    def tranpose_inputs(self, data, model_config):
        '''Expand start and end years to timeseries'''

        data = data.copy()
        
        # get labels
        label = model_config['label']
        time_label = model_config['time_label']
        interp_label = model_config['interpolation_label']
        start = model_config['start_time']
        end = model_config['end_time']
        start_var = model_config['start_variable']
        end_var = model_config['end_variable']

        config_idx = [interp_label, start, end, start_var, end_var]
        
        # for each row in dataframe interpolate values for date range by given method
        idat = [self.interpolate_ramp(r[0], r[1], r[2], r[3], r[4]) for r in data[config_idx].values]

        data[start] = pd.to_datetime(data[start], format='%Y')
        data[end] = pd.to_datetime(data[end], format='%Y')

        # create standard date ranges
        dates = [pd.date_range(r[0], r[1], freq='AS') for r in data[[start, end]].values]
        # reshape data frame and load with range and probabilty interpolation
        lens = [len(x) for x in dates]

        data_l = pd.DataFrame({col:np.repeat(data[col].values, lens) for col in data})
        data_l[time_label] = pd.to_datetime(np.concatenate(dates)).year
        data_l[label] = np.concatenate(idat)
        return data_l.drop([start, end, start_var, end_var], axis=1)
    
    def interpolate_ramp(self, ramp_type, start_year, end_year, start_val, end_val):
        """
        Will do a linear or log scale interoplation
        """
        interp_funcs = {'linear': np.linspace, 'log': np.geomspace, 'exp_decay': self.exp_decay, 'geom_shifted': self.geom_shifted}
        return interp_funcs[ramp_type](start_val, end_val, (end_year - start_year + 1))
    
    def exp_decay(self, start_val, end_val, window):
        # decay constant is set so that lower limit is reached in final period
        decay_const = window * ( 0.2/ 21)
        exp_decay = (start_val - end_val) * np.exp(-decay_const * np.arange(0, window))
        return start_val + np.cumsum(np.append(0, np.diff(exp_decay)))
    
    def geom_shifted(self, start_val, end_val, window, n=5):
        # creates a geometric ramp shifted back n_periods
        shift = np.zeros(n) + start_val
        geom = np.geomspace(start_val, end_val, window)
        return np.concatenate((shift, geom))[:-n]

    def equipment_calc(self, stock):
        '''Implements stock turnover calculation'''
        stock = stock.sort_values(by=list(self.equipment_sort_index))
        
        stock[self.equipment_label] = stock[list(self.equipment_calc_variables)].cumprod(axis=1).iloc[:,-1]
              
        # will hold turnover calculations
        turnover_list = []
        
        # loop over groups to calculate stock turnover for each equipment type
        for idx, grp in stock.groupby(list(self.equipment_calc_index)):
            # call function to calculate equipment turnover
#             if idx == ('Residential', 'Single Family', 'New', 'Lighting Interior Specialty'):
#                 print('debug')

            turnover = self.turnover_calc(
                grp[self.equipment_label].values,
                grp[self.effective_useful_life_label].values,
                grp[self.efficiency_level_label].values,
                grp[self.ramp_efficiency_level_label].values,
                grp[self.ramp_efficiency_probability_label].values,
            )

            turnover_list.append(turnover)

        # convert list of equipment turnovers to matrix and merge to dataframe
        stock[self.equipment_stock_label] = np.hstack(turnover_list)      
        
        # calculate energy estimate
        stock[self.equipment_consumption_label] = (stock[self.equipment_stock_label] * stock[self.equipment_measure_consumption_label]) / 1000
        return stock.drop([self.equipment_label], axis=1)
    
    def turnover_calc(self, stock, eul, e_level, se_level, ramp_prob):
        """
        Matrix calculation to estimate turnover by end use
        """
        # get count of efficiency type
        e_levels = np.unique(e_level).shape[0]
        # these matrices will hold total stock and turnover
        s_mat = np.zeros([np.int(len(stock)/e_levels), e_levels])
        t_mat = np.zeros([np.int(len(stock)/e_levels), e_levels])
        s_mat_new = np.zeros(e_levels)
        # reshape input matrices for useful life and efficiency levels
        stock_mat = np.reshape(stock, np.flip(s_mat.shape)).T
        eul_mat = np.reshape(eul, np.flip(s_mat.shape)).T
        e_mat = np.reshape(e_level, np.flip(s_mat.shape)).T
        se_mat = np.reshape(se_level, np.flip(s_mat.shape)).T
        prob_mat = np.reshape(ramp_prob, np.flip(s_mat.shape)).T

        # set initial stock values
        s_mat[0, :] = np.rint(stock_mat[0, :])

        # loop over each forecast year [i]
        for i in range(s_mat.shape[0] - 1):
            # loop over each efficiency type [j]
            t_mat[i + 1, :] = np.rint(s_mat[i, :]/eul_mat[i, :])
            # check if there is any change in stock from new buildings or saturation changes
            s_mat_new = np.diff(np.rint(stock_mat), axis=0)[i]
            for j in range(s_mat.shape[1]):
                # if efficiency type [j] is less than standard then convert to standard efficiency type
                if se_mat[i + 1, j] >  e_mat[i, j]:
                    offset = np.int(se_mat[i + 1, j] - e_mat[i, j])
                    # need to make sure that transition probably is factored in when accounting for new building stock
                    # this calculates the amount of below standard stock that will not turnover
                    s_mat[i + 1, j] = np.rint((s_mat[i, j] - t_mat[i + 1, j] * prob_mat[i + 1, j]) + s_mat_new[j] * (1 - prob_mat[i + 1, j])) 
                    # this calculates the amount of below standard stock that will turnover to higher efficiency
                    s_mat[i + 1, j + offset] = np.rint((s_mat[i + 1, j + offset] + t_mat[i + 1, j] * prob_mat[i + 1, j] + prob_mat[i + 1, j] * s_mat_new[j]))
                # if efficiency type [j] is above standard then replace with itself 
                else:
                    s_mat[i + 1, j] = s_mat[i, j] + s_mat[i + 1, j] + s_mat_new[j]
        
        return np.ravel(s_mat, order='F')