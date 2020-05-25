# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import logging
import psycopg2 as ppg  # type: ignore
from dbfread import DBF  # type: ignore

import types
from typing import List, Set, Dict, Tuple, Optional, Any

module_logger = logging.getLogger('csv_to_table_test')
module_logger.setLevel(logging.INFO)
module_logger.info('Hi from module logger')


class csv_to_table(object):
    def __init__(self):
        self.logger = module_logger  # logging.getLogger('csv2table')
        self.logger.info('csv_to_table initialized')

        self.quote_string: str = '$q$'
        self.maxint16: int = 2 ** (16 - 1) - 1
        self.maxint32: int = 2 ** (32 - 1) - 1

        self.type_dict = {0: 'varchar', 1: 'float', 2: 'bigint', 3: 'integer', 4: 'smallint', 5: 'date', 6: 'time'}

        return

    def test_connection(self, host: str, port: str, db: str, user: str, pw: str):
        """
        Tests the connection to a postgres database. Raises an exception when
        the connection could not be made.

        Args:
            host (str): host name where the server resides.
            port (str): port of Postgres
            db (str): name of the database
            user (str): user name
            pw (str): password

        Returns:
            None
        """
        try:
            connection=ppg.connect(user=user,
                                   password=pw,
                                   host=host,
                                   port=port,
                                   database=db)
            cursor = connection.cursor()
            self.logger.info(str(connection.get_dsn_parameters()))

            cursor.execute('select version();')
            record = cursor.fetchone()
            self.logger.info('Connected to:' + str(record))

        except (Exception, ppg.Error) as error:
            self.logger.error("Error while connecting to PostgreSQL" + str(error))

        finally:
            # closing database connection.
            if(connection):
                # cursor.close()
                # connection.close()
                # self.logger.info("PostgreSQL connection is closed")

                return connection, cursor
            else:
                return None

    def pg_creds(self, credential_file: str, server: str, port: str):
        creds = []
        search: str = server + ':' + port

        with open(credential_file, 'r') as cache:
            for line in cache:
                creds.append(line.strip())
                found = line.find(search, 0)
                if found == 0:
                    parts = line.split(':')

                    return parts[3], parts[4].rstrip()

        return None, None

    def read_table(self, credentials, host, database, table, port='5432',
                   columns='*', where=None):
        """
        Reads selected columns from a table from a postgres database and
        returns it as a pandas dataframe.

        Args:
            cred_file (str): Path to the file containing user and passsword
            host (str): Name of host where the postgres database is hosted
            database (str): Name of the database containing the table
            table (table): Name of the table to fetch
            port (str): Port of the postgres database, defaults top 5432
            columns (str): Columns to fetch from table, default is * (all)

        Returns:
            dat (pandas.DataFrame):

        """

        try:
            if type(credentials) is tuple:
                (username, password) = credentials
            else:
                username, password = self.pg_creds(credentials, host, port)

            sql = 'SELECT ' + columns + ' FROM ' + table + ';'

            try:
                connection, cursor = self.test_connection(user=username, pw=password,
                                                          host=host, port=port, db=database)

                # Fetch the table from postgres
                cursor.execute(sql, table)
                result = cursor.fetchall()

                # Build SQL string
                sql = 'SELECT ' + columns + ' FROM ' + table
                if where is not None:
                    sql += ' WHERE ' + where
                sql += ' LIMIT 0;'

                # Fetch column names of the table
                cursor.execute(sql)
                colnames = [desc[0] for desc in cursor.description]

            finally:
                if (connection):
                    cursor.close()
                    connection.close()
        finally:
            # Always destroy username and password
            user = 'user'
            password = user

        # Convert sql result to dataframe
        dat = pd.DataFrame(result)

        # Assign columns to dataframe
        dat.columns = colnames

        return dat

    def which_type(self, x: str) -> int:
        """
        Determines the type of a string.

        Args:
            x (str): value of which the type should be determined.

        Returns:
            int: 0 = string, 1 = float, 2 = int64, 3 = int32, 4 = int16,
                 -1 = empty = missing data


        """
        x = x.strip()
        if len(x) == 0:  # missing data/null
            return -1

        try:
            a = float(x)
            try:
                b = int(a)
            except ValueError:
                return 1  # is floating point
            else:
                if a == b:  # is integer, 32 or 64 bit
                    if (a < -self.maxint32) and (a > self.maxint32):
                        return 2  # is int64
                    if (a < -self.maxint16) or (a > self.maxint16):  # test if integer is int64
                        return 3  # int32
                    else:
                        return 4  # int16

        except ValueError:
            return 0  # x is string

        return 0

    def determine_csv_column_type(self, col: pd.Series):  # -> (int, int, bool):
        """
        Find out which type of column the col is,

        When it is a string, the maximum length of the string
        is added, when it contains missing data (empty string
        or nan) True is returned

        Args:
            col (pd.Series): column to determine the type

        Returns:
            data_type (int): 0 = str, 1 = float, 2 = int64, 3 = int32.
            data_len (int): maximum length of string when type is 0.
            misdat (bool): True if col contains missing data, else False.
        """

        data_type: int = 0  # 0 = str, 1 = float, 2 = int64, 3 = int32
        data_len: int = 0
        misdat: int = 0
        typ: int = 0

        self.logger.debug('\nDetermining column' + str(col.name) + 'Current type is' + str(col.dtype))
        if col.dtype == 'float' or col.dtype == 'float64':
            data_type = 1
            self.logger.debug('col data type = ' + str(data_type))
            # TODO: Compute misdat
        elif col.dtype == 'int' or col.dtype == 'int64':
            ma = col.max()
            mi = col.min()
            if (mi < -self.maxint32) or (ma > self.maxint32):  # test if integer is int64
                data_type = 2
            elif (mi < -self.maxint16) or (ma > self.maxint16):  # test if integer is int32
                data_type = 3
            else:
                data_type = 4

            self.logger.debug('col data type = ' + str(data_type))

        else:
            self.logger.debug('col data type = ' + str(data_type) + ' / ' + str(col.dtype))
            for val in col:
                self.logger.debug('col data type = ' + str(val) + ': ' +
                                  str(data_type) + ' / ' + str(col.dtype))
                if pd.isnull(val):
                    typ = -1
                else:
                    val = str(val)
                    if len(val) > data_len:
                        data_len = len(val)
                    typ = self.which_type(val)

                if typ == -1:
                    misdat += 1
                else:
                    if typ < data_type:
                        data_type = typ

                self.logger.debug('Data type is: ' + str(data_type) +
                                  ', misdat = ' + str(misdat))

        return data_type, data_len, misdat

    def determine_dbf_column_type(self, col: pd.Series, missing_val):
        data_type: int = 0  # 0 = str, 1 = float, 2 = int64, 3 = int32
        data_len: int = 0
        misdat: int = 0
        typ: int = 0

        self.logger.debug('Determining column ' + str(col.name))
        if col.dtype == 'O':
            data_len = col.str.len().max()
        elif col.dtype == 'float' or col.dtype == 'float64':
            data_type = 1
            col = col.replace(float(missing_val), np.nan)
            self.logger.debug('col data type = ' + str(data_type))
            # TODO: Compute misdat
        elif col.dtype == 'int' or col.dtype == 'int64':
            ix = col == int(missing_val)
            temp = col[~ix]
            misdat = len(col) - len(temp)
            ma = temp.max()
            mi = temp.min()
            if (mi < -self.maxint32) or (ma > self.maxint32):  # test if integer is int64
                data_type = 2
            elif (mi < -self.maxint16) or (ma > self.maxint16):  # test if integer is int32
                data_type = 3
            else:
                data_type = 4

            col = col.apply(str)
            col[ix] = ''

        else:
            for val in col:
                if pd.isnull(val):
                    typ = -1
                else:
                    if len(val) > data_len:
                        data_len = len(val)
                    typ = self.which_type(val)

                if typ == -1:
                    misdat += 1
                else:
                    if typ < data_type:
                        data_type = typ

                self.logger.debug('Data type is: ' + str(data_type) +
                                  ', misdat = ' + str(misdat))

        return col, data_type, data_len, misdat

    def convert_data_frame(self, data: Any):
        """
        Analyzes a dataframe and assigns a type to each column.

        Args:
            data (Any): DESCRIPTION.

        Returns:
            data (DataFrame): data as input.
            data_types (array of int): for each column of data its type.
            data_lens (array of int): for each column, when text then max length, else 0.
            missing_data (array of int): for each column whether it contains missing data.

        """
        n_rows, n_cols = data.shape

        data_types = np.array(np.zeros(n_cols)).astype(np.int)
        data_lens = np.array(np.zeros(n_cols)).astype(np.int)
        missing_data = np.array(np.zeros(n_cols)).astype(np.int)

        for i, c in enumerate(data.columns):
            data_types[i], data_lens[i], missing_data[i] = self.determine_csv_column_type(data[c])

        return data, data_types, data_lens, missing_data

    def read_simple_file(self, filename):
        with open(filename, 'r') as infile:
            lines = infile.readlines()

        lines = [line[:-1] for line in lines]

        return lines

    def read_csv_file(self, fn: str, encoding: str='UTF-8', sep=','):
        """
        Reads a csv file and returns it as a DataFrame.

        Args:
            fn (str): file name of the csv file.
            encoding (str, optional): Character encoding of the file. Defaults to 'UTF-8'.

        Returns:
            data (DataFrame): data as input.
            data_types (array of int): for each column of data its type.
            data_lens (array of int): for each column, when text then max length, else 0.
            missing_data (array of int): for each column whether it contains missing data.

        """
        data = pd.read_csv(fn, encoding=encoding, low_memory=False, sep=sep, index_col=False)
        data, data_types, data_lens, missing_data = self.convert_data_frame(data)

        return data, data_types, data_lens, missing_data

    def read_csv_files(self, front: str, rear: str, between: List[str],
                       encoding: str='UTF-8', sep=',', limit=None):
        """
        Reads a list of csv files and concatenates them as one DataFrame.

        Args:
            front (str): front part of the file name.
            rear (str): rear part of the file name.
            between (List[str]): list of texts that should all be concatenated
                with front and rear to form a list of file names.
            encoding (str, optional): encoding of all files. Defaults to 'UTF-8'.
            limit (str, optional): For each file the maximum number of rows to be read,
                when None all is read. Defaults to None.

        Returns:
            data (DataFrame): data as input.
            data_types (array of int): for each column of data its type.
            data_lens (array of int): for each column, when text then max length, else 0.
            missing_data (array of int): for each column whether it contains missing data.

        """
        names: Dict[str, Tuple[str, Any]] = {}

        for year in between:
            filename = front + year + rear
            data = pd.read_csv(filename, encoding=encoding, low_memory=False, nrows=1)
            names[year] = (filename, data.columns)

        # Check whether all CSV's have the same headers
        result = True
        start = between[0]
        headers_start = names[start][1]
        self.logger.info('Comparing headers to: ' + str(start))
        for year in between[1:]:
            result = True
            headers = names[year][1]

            # Check for eaual length
            if len(headers) != len(headers_start):
                raise ValueError(str(year) + ': not same amount of headers:' +
                                 str(len(headers)) + ' vs ' + str(len(headers_start)))
            else:
                # Check for equal position of headers
                for hdr_1, hdr_2 in zip(headers_start, headers):
                    if hdr_1 != hdr_2:
                        self.logger.warn(str(year) + ': Headers not equal position: ' +
                                         str(hdr_1) + ' <> ' + str(hdr_2))
                        # self.logger.warn(start, headers_start)
                        # self.logger.warn(year, headers)

                        result = False
                        break

                # if above not succesfull, are headers just mixed up?
                if not result:
                    result = True
                    for hdr in headers:
                        if hdr not in headers_start:
                            result = False
                            self.logger.warn(str(year) + ' Not found header: ' + str(hdr))

                    if result:
                        self.logger.info(str(year) + ': All headers are found in' + str(start))

                    for hdr in headers_start:
                        if hdr not in headers:
                            result = False
                            self.logger.warn(str(start) + ' Not found header: ' + str(hdr))

                    if result:
                        self.logger.info(start, ': All headers are found in', year)

                else:
                    self.logger.info(str(year) + ': Headers are equal')

        # Abort when columns do not match
        if not result:
            raise ValueError('Columns do not match sufficiently')

        # unify all data
        first_frame = True
        df: Any = None

        for year in between:
            filename = names[year][0]
            print(filename)
            data = pd.read_csv(filename, encoding=encoding, low_memory=False,
                               sep=sep, nrows=limit)
            data.replace(to_replace='.', value='', inplace=True)
            data.insert(0, 'Jaar', year, True)
            if first_frame:
                df = data
                first_frame = False
            else:
                df = pd.concat([df, data])

        # df = df.set_index('OngevalID')

        data, data_types, data_lens, missing_data = self.convert_data_frame(df)

        return data, data_types, data_lens, missing_data

    def write_csv_file(self, data, data_types, data_lens, misdat, table: str,
                       csv_dir: str='/tmp', sql_dir: str='', sep: str=',',
                       primary=None, encoding: str='UTF-8',
                       pnt_dest: str='geo_lokatie', pnt_src: str='latitude, longitude') -> None:
        """


        Args:
            data (DataFrame): data frame conatining the data.
            data_types (list): types for each column.
            data_lens (list): max length for each column when text.
            misdat (list): DESCRIPTION.
            table (str): name of the table to be created.
            csv_dir (str): directory to save the table values as a .csv file into
            sql_dir (str): directory in which to save the sql file
            sep (str, optional): CSV separator. Defaults to ','.
            primary (str, optional): name of column containing primary key.
                                      Defaults to None.
            pnt_dest (str, optional): point destination column when not None.
                                      Defaults to 'lokatie'.
            pnt_src (str, optional): source for the point when destination not None.
                                      Defaults to 'latitude, longitude'.

        Returns:
            None: DESCRIPTION.

        """
        temp_name: str = os.path.join(csv_dir, table) + '.csv'
        table_def: str = self.build_headers(data, data_types, data_lens, misdat,
                                            table, primary=primary)
        table_def += "\copy " + table + " FROM '" + temp_name + \
                     "' DELIMITER '" + sep + "' CSV HEADER;\n\n"

        if pnt_dest is not None:
            table_def += self.add_point(table, pnt_dest, src=pnt_src)

        with open(os.path.join(sql_dir, table) + '.sql', 'w') as file_handle:
            file_handle.write(table_def)

        data.to_csv(temp_name, header=True, index=False, sep=sep, encoding=encoding)

        return None

    def read_dbf_file(self, fn: str, misdat=None):
        table = DBF(fn, encoding='UTF-8')
        data = pd.DataFrame(iter(table))
        ix = data['WATER'] != 'NEE'
        data = data[~ix]

        # data = data[['WATER', 'P_STADVERW', 'AUTO_LAND', 'AUTO_HH']]
        n_rows, n_cols = data.shape

        data_types = np.array(np.zeros(n_cols)).astype(np.int)
        data_lens = np.array(np.zeros(n_cols)).astype(np.int)
        missing_data = np.array(np.zeros(n_cols)).astype(np.int)

        temp = pd.DataFrame(index=data.index)
        for i, c in enumerate(data.columns):
            col, data_types[i], data_lens[i], missing_data[i] = self.determine_dbf_column_type(data[c], misdat)
            temp = temp.join(col)

        return temp, data_types, data_lens, missing_data

    def describe_data(self, data, data_types, data_lens, missing_data) -> None:
        data_description = pd.DataFrame(index=data.columns,
                                        columns=['Data Types', 'Data Lengths', 'Missing'])
        data_description.loc[:, 'Data Types'] = data_types
        data_description.loc[:, 'Data Lengths'] = data_lens
        data_description.loc[:, 'Missing'] = missing_data

        self.logger.info(str(data_description))

        return

    def build_headers(self, data, data_types, data_lens, misdat, table: str, primary=None) -> str:
        build: str = 'DROP TABLE IF EXISTS ' + table + ';\n\n'
        build += 'CREATE TABLE ' + table + '(\n'
        for i, col in enumerate(data.columns):
            if data_types[i] == 0:
                typ = self.type_dict[0] + '(' + str(int(data_lens[i])) + ')'
            else:
                typ = self.type_dict[data_types[i]]

            col = str(col).strip().replace(' ', '_')
            build += '    ' + str(col).strip() + ' ' + typ
            if primary is not None and col == primary:
                build += ' PRIMARY KEY'

            if col != data.columns[-1]:
                build += ',\n'
            else:
                build += '\n    );\n\n'

        ### for

        return build

    def build_string(self, data, data_types, data_lens, misdat, table: str, primary=None):
        minval = np.array(np.zeros(len(data.columns)))
        maxval = np.array(np.zeros(len(data.columns)))
        """
        build: str = 'DROP TABLE IF EXISTS ' + table + ';\n\n'
        build += 'CREATE TABLE ' + table + '(\n'
        for i, col in enumerate(data.columns):
            if data_types[i] == 0:
                typ = self.type_dict[0] + '(' + str(int(data_lens[i])) + ')'
            else:
                typ = self.type_dict[data_types[i]]

            build += '    ' + str(col).strip() + ' ' + typ
            if not primary is None and col == primary:
                build += ' PRIMARY KEY'

            if col != data.columns[-1]:
                build += ',\n'
            else:
                build += '\n    );\n\n'

        ### for
        """
        build = self.build_headers(data, data_types, data_lens, misdat, table, primary=primary)

        counter: int = 0
        self.logger.info('Creating ' + str(len(data)) + ' records')
        build += '\nINSERT INTO ' + table + '\nVALUES\n'
        for index, row in data.iterrows():
            if counter % 1000 == 0:
                print(counter, index)

            counter += 1

            build += '('
            for i, col in enumerate(data.columns):
                if pd.isnull(row[col]):
                    value = 'DEFAULT'
                else:
                    value = str(row[col]).strip()
                    if len(value) == 0:
                        value = 'DEFAULT'

                if value != 'DEFAULT':
                    if data_types[i] == 0:
                        value = self.quote_string + value + self.quote_string
                    else:
                        x = float(value)
                        if x < minval[i]:
                            minval[i] = x
                        if x > maxval[i]:
                            maxval[i] = x

                build += value
                if col != data.columns[-1]:
                    build += ','

            build += '),\n'
        ### for

        build = build[:len(build) -2] + ';\n\n'

        return build

    def csv_to_string(self, fn: str, table: str, primary: str=None, encoding='UTF-8'):
        self.logger.info('Reading ' + fn)
        data, data_types, data_lens, data_miss = \
            self.read_csv_file(fn, encoding=encoding)
        self.logger.info(str(len(data)) + ' rows counted\n')

        self.describe_data(data, data_types, data_lens, data_miss)
        self.logger.info('Creating ' + str(table))
        string = self.build_string(data, data_types, data_lens, data_miss, table, primary)

        return string

    def dbf_to_string(self, fn: str, table: str, primary: str=None, misdat=None):
        self.logger.info('Reading ' + fn)
        data, data_types, data_lens, data_miss = self.read_dbf_file(fn, misdat)
        self.logger.info(str(len(data)) + ' rows counted\n')

        self.describe_data(data, data_types, data_lens, data_miss)
        self.logger.info('Creating ' + table)
        string = self.build_string(data, data_types, data_lens, data_miss, table, primary)

        return string

    def add_point(self, table: str, column: str, src: str='latitude, longitude', srid: str='4326'):
        string: str = 'ALTER TABLE ' + table + ' ADD COLUMN '
        string += column + ' geometry(Point, 4326);\n'
        string += 'UPDATE ' + table + ' SET ' + column + '=st_SetSrid(st_MakePoint('
        string += src + '), ' + srid + ');\n\n'

        return string

    def write_table(self, string, fn):
        with open(fn, 'w') as f:
            f.write(string)

        self.logger.info('Written ' + fn + ' records')

    def transfer_table(self, string, host: str, port: str, db: str, user: str, pw: str) -> None:
        try:
            connection = ppg.connect(user=user,
                                     password=pw,
                                     host=host,
                                     port=port,
                                     database=db)

            try:
                cursor = connection.cursor()
                self.logger.info(str(connection.get_dsn_parameters()))

                cursor.execute('select version();')
                record = cursor.fetchone()
                self.logger.info('Connected to: ' + str(record))

                cursor.execute(string)
                connection.commit()
                try:
                    record = cursor.fetchall()
                    self.logger.info('Result: ' + str(record))

                except ppg.ProgrammingError as error:
                    self.logger.warn('No info returned by query: ' + str(error))

            finally:
                cursor.close()

        except (Exception, ppg.Error) as error:
            self.logger.error("Error while connecting to PostgreSQL: " + str(error))

        finally:
            # closing database connection.
            if(connection):
                cursor.close()
                connection.close()
                self.logger.info("PostgreSQL connection is closed")

        return
