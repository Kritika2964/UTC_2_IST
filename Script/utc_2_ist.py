
import pytesseract
import numpy as np
import cv2
import imageio
import dateutil.parser
import datetime
import pandas as pd
import os
import re
import pickle     
import ftplib
import csv
import time
# Configuration for pytesseract

# Path to the Tesseract OCR executable.
pytesseract.pytesseract.tesseract_cmd = r''

# Path to the custom trained data file located in the Config directory.
custom_model_path = '' 

class UtcConverter:
    def __init__(self,
                 # Use of FTP server only if downloading from website. 
                 ftpwaitsec=300,  # Time in seconds to wait between FTP checks.
                 ftp_server='',    # FTP server address.
                 username='',      # FTP server username.
                 password='',      # FTP server password.
                 remote_dir='',    # Remote directory on the FTP server containing the RADAR images.

                 # Use of local file if not downloading from website.
                 local_dir='',     # Local directory path where the RADAR images will be downloaded. This is the Input directory.
                 config_dir='',    # Local directory path to the Config directory containing necessary CSV files and trained data.
                 patterns=['caz_', 'ppi_', 'sri_', 'ppz_', 'pac_', 'ppv_', 'vp2_'],  # List of filename patterns to process.
                 log_path='',      # Path to the Log directory where logs and preprocessed images will be stored.
                 op_path=''        # Path to the Output directory where the processed images with IST timestamps will be saved.
                    
                    #     r'^caz_*\.gif$',  # Matches files that start with 'caz_' and end with '.gif'
                    #     r'^ppi_*\.gif$',  # Matches files that start with 'ppi_' and end with '.gif'
                    #     r'^sri_*\.gif$',  # Matches files that start with 'sri_' and end with '.gif'
                    #     r'^ppz_*\.gif$',  # Matches files that start with 'ppz_' and end with '.gif'
                    #     r'^pac_*\.gif$',  # Matches files that start with 'pac_' and end with '.gif'
                    #     r'^ppv_*\.gif$',  # Matches files that start with 'ppv_' and end with '.gif'
                    #     r'^vp2_*\.gif$',  # Matches files that start with 'vp2_' and end with '.gif'
                    # 
                 ) -> None:
        self.ftp_server = ftp_server
        self.username = username
        self.password = password
        self.remote_dir = remote_dir
        self.local_dir = local_dir
        self.config_dir = config_dir
        self.patterns = patterns
        self.processed_csv = f'{config_dir}/lastprocessed.csv'
        self.ftpwaitsec = ftpwaitsec
        self.parameter = pd.read_csv(f'{config_dir}/Parameter.csv')
        self.stationlist = pd.read_csv(f'{config_dir}/Stations.csv')
        self.log_path = log_path
        self.errorlogfile = f'{self.log_path}/errorlog.csv'
        self.timestamp = str(datetime.datetime.now())
        self.date = str(datetime.date.today())
        self.timestamp1 = self.timestamp.replace(':','_').replace(' ','_')
        self.op_path = f'{op_path}/{self.timestamp1}'
        self.successlogfile = f'{self.log_path}/{self.date}.pickle'

        
        #print(self.parameter, self.parameter.info(), self.stationlist, self.stationlist.info()) 
        #merged = self.stationlist.join(self.parameter,on='Config_id')
        # Convert Config_id to string (object) in both DataFrames
        self.parameter['Config_id'] = self.parameter['Config_id'].astype(str)
        self.stationlist['Config_id'] = self.stationlist['Config_id'].astype(str)

        # Now perform the join
        self.configstn = self.stationlist.join(self.parameter.set_index('Config_id'), on='Config_id')
        
        
        # Ensure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        
        # Ensure the local directory exists
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if not os.path.exists(self.op_path):
            os.makedirs(self.op_path)
    
    def readprocessed(self):
        processed_files = {}
        if os.path.exists(self.processed_csv):
            with open(self.processed_csv, mode='r') as f:
                reader = csv.reader(f)
                
                processed_files = {rows[0]: datetime.datetime.strptime(rows[1], '%Y-%m-%d %H:%M:%S') for rows in reader}
        return processed_files

    def get_ftp_file_mtime(self, file):
        resp = self.ftp.sendcmd(f'MDTM {file}')
        mtime = datetime.datetime.strptime(resp[4:], "%Y%m%d%H%M%S")
        return mtime
 
    def download(self):
        files = os.listdir(self.local_dir)
        files = [f'{self.local_dir}/{a}' for a in files]
        for file in files:
            os.remove(file)
        # Connect to the FTP server
        self.ftp = ftplib.FTP(self.ftp_server)
        self.ftp.login(user=self.username, passwd=self.password)
        self.ftp.cwd(self.remote_dir)
        
        # List files in the remote directory
        files = self.ftp.nlst()
        processed_files = self.readprocessed()
        toProcess = []
        #print(files)
        for file in files:
            check = file[0:4]
            #print(check,check.lower() in self.patterns)
            if check.lower() in self.patterns:
                toProcess.append(file)
        toProcess.sort()
        # Download files matching any of the patterns
        for file in toProcess:
            mtime = self.get_ftp_file_mtime(file)
            if file not in processed_files or (mtime - processed_files[file]).total_seconds() >= self.ftpwaitsec:
                check1 = self.get_ftp_file_mtime(file)
                check2 = self.ftp.size(file)
                
                # Wait for a short interval and then check size and mtime again
                print('waiting')
                time.sleep(0.25)  # Adjust the sleep time as needed
                new_check1 = self.get_ftp_file_mtime(file)
                new_check2 = self.ftp.size(file)
                print('checking')
                
                # Check if the file is still being written to
                if (check1 == new_check1 and check2 == new_check2):
                    #print('Downloading ', file)
                    local_file_path = os.path.join(self.local_dir, file)
                    with open(local_file_path, 'wb') as local_file:
                        self.ftp.retrbinary(f'RETR {file}', local_file.write)
                    print(f'Downloaded: {file}')
                    processed_files[file] = mtime
                else:
                    print(f'{file} is being written... so skipping to download now')
            else:
                print(file, mtime,(mtime - processed_files[file]).total_seconds() )
            
            
        # Close the FTP connection
        self.ftp.quit()
        # Save the processed files with updated timestamps back to processed.csv
        with open(self.processed_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            for file, mtime in processed_files.items():
                writer.writerow([file, mtime.strftime('%Y-%m-%d %H:%M:%S')])
    
    def writelog(self, filename, message):
        with open(filename,'a') as f:
            f.write(message)

    def successlog(self, Filename, StationID,ProductID,ConfigID,Input_img,Extracted_text,Status):
        if os.path.exists(self.successlogfile):
            log = pd.read_pickle(self.successlogfile)
            
        else:
            log = pd.DataFrame(columns=['Timestamp','Filename','StationID','ProductID','ConfigID','Input_img','Extracted_text','IST_Info','Status'])
        index = log.shape[0]
        info = pd.DataFrame({'Timestamp':[self.timestamp],
                            'Filename': [Filename],
                            'StationID': [StationID],
                            'ProductID':[ProductID],
                            'ConfigID': [ConfigID],
                            'Input_img': [Input_img],
                            'Extracted_text': [Extracted_text],
                            'Status': [Status]

                            }, index=[index])
        log = pd.concat([log,info])
        log.to_pickle(self.successlogfile)


    def readimage(self, filepath):
        try:
            gif = imageio.mimread(filepath)[0]
        except:
            return None, f'NO Frames found in GIF file or Invalid GIF file - {filepath}'

        img = cv2.cvtColor(gif, cv2.COLOR_RGB2BGR)
        original_img = img.copy()
        return img, 'Success'

    
    def preprocess_image(self):
        """
            Carry out preprocessing to improve the accuracy of the image to text conversion"""
        
        try:
            image = self.original_img.copy()
            cropped_image = image[self.y_start:self.y_end, self.x_start:self.x_end]

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            cropped_image = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # Added: Gaussian blur
            if self.apply_gaussian_blur:
                cropped_image = cv2.GaussianBlur(cropped_image, (1, 1), cv2.BORDER_DEFAULT)

            #Changed: Adjust the scale_percent as needed
            if cropped_image.shape[0] < 50 or cropped_image.shape[1] < 100:
                scale_percent = 200
            else:
                scale_percent = 150

            width = int(cropped_image.shape[1] * scale_percent / 100)
            height = int(cropped_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            #self.preprocessedimg = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)
            self.preprocessedimg = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


            return 'Success'
        except Exception as e:
            return 'Preprocessing image failed'

    def preprocesstxt(self, text):
        print(text, 'RAW TEXT')
        text = text.replace('\n', ' ')

        # To remove unwanted unicode characters, if any detected
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        text = text.translate(translator)
        #print(text)

        #Added: Post-process the OCR result to correct common errors
        text = re.sub(r'(\d{2}:\d{2}:\d{2})\d', r'\1', text)  # Remove extra digits from time
        text = text.replace(":", ":", 2)  # Ensure only two colons in the time
        text = text.replace('z', '').replace('Z', '').replace('(','').replace(')','').replace('\\','').replace('|','').replace('I','').replace('{','').replace('}','')

        # Added: Correct misread times like "94:10:01" to "04:10:01"
        # This assumes misreads result in a leading digit error, common in OCR
        parts = text.split()
        if len(parts) > 0 and len(parts[0]) == 8:  # Check if the first part is time and misread
            hours, rest = parts[0].split(":", 1)
            if int(hours) > 23:  # If hours exceed 23, correct it
                hours = "0" + hours[1:]  # Example correction: 94 -> 04
            corrected_time = f"{hours}:{rest}"
            text = corrected_time + " " + " ".join(parts[1:])

        # Added: Handle "Date:04/08/2023 Time:10:36:43 UTC" format
        if "Date:" in text and "Time:" in text:
            date_part = text.split("Date:")[1].split("Time:")[0].strip()
            time_part = text.split("Time:")[1].strip()
            text = f"{time_part} {date_part}"

        # Added: Handle "10:36:43 4 AUG 2023" format
        if len(parts) == 2 and re.match(r'^\d{2}:\d{2}:\d{2}$', parts[0]) and re.match(r'^\d{1,2} \w+ \d{4}$', parts[1]):
            text = f"{parts[0]} {parts[1]}"

        # Added: To handle / replacemnt
        for char in ['!', 'f', 'F']:
            text = text.replace(char, '')
        original_text = text
        try:
            # Preprocess the text
            # Split the time part and fix the seconds component
            time_part, date_part = text.split(' ', 1)
            hours, minutes, seconds = time_part.split(':')
            seconds = seconds[:2]  # Take only the first two digits of seconds
            fixed_time_part = f'{hours}:{minutes}:{seconds}'
            # Combine fixed time part with the rest of the text
            text = f'{fixed_time_part} {date_part}'
        except:
            text = original_text
        

        return text
    
    def extract_time_text(self):
        """Image to text conversion and post process the text"""
        #image = self.preprocessedimg
        #custom_config = f'--tessdata-dir /home/irad2025/AGR/UTC2IST/Data/CONFIG -l eng'
        #text = pytesseract.image_to_string(image, config=custom_config)
        #print(text)
        try:
            image = self.preprocessedimg

            #Added: custom_config
            custom_config = r'--oem 3 --psm 7'
            #custom_config = f'--tessdata-dir /home/irad2025/AGR/UTC2IST/Data/CONFIG -l eng'

            text = pytesseract.image_to_string(image, config=custom_config)
            text = self.preprocesstxt(text)
            custom_model = False
            try:
                check1 = dateutil.parser.parse(text)    
            except:
                custom_model = True

            if custom_model:
                custom_config = f'--tessdata-dir /home/irad2025/AGR/UTC2IST/Data/CONFIG -l eng'
                text = pytesseract.image_to_string(image, config=custom_config)
                try:
                    check1 = dateutil.parser.parse(text)    
                    custom_model = False
                except:
                    custom_model = True
            
            if custom_model:
                custom_config = f'--tessdata-dir /home/irad2025/AGR/UTC2IST/Data/CONFIG -l eng_best'
                text = pytesseract.image_to_string(image, config=custom_config)
                try:
                    check1 = dateutil.parser.parse(text)    
                    custom_model = False
                except:
                    custom_model = True
            
            if custom_model:
                custom_config = f'--tessdata-dir /home/irad2025/AGR/UTC2IST/Data/CONFIG -l eng_fast'
                text = pytesseract.image_to_string(image, config=custom_config)
                

            text = self.preprocesstxt(text)
            print(text, ' -  Final Converted text')
            self.convertedtxt = text
            return 'Success'
        except Exception as e:
            return f"Error extracting text from image: {e}"

    def convert_to_ist(self):
        """Convert extracted text to IST"""
        try:
            text = self.convertedtxt
            UTC = dateutil.parser.parse(text)
            IST = UTC + datetime.timedelta(hours=5, minutes=30)
            self.convertedtxt = IST
            #print(f"Extracted UTC time: {UTC}, Converted IST time: {IST} \n")
            return 'Success'
        except (ValueError, dateutil.parser._parser.ParserError) as e:
            return f"Failed to parse time from text: {text} ({e}) \n"
        
    
    def overlay_text_on_image(self):
        """Add the information back to the Image"""
        try:
            image = self.original_img
            IST = self.convertedtxt
            org = self.org
            twoline = self.twoline
            high_res = self.high_res
            font = cv2.FONT_HERSHEY_PLAIN
            fontScale = 2.5 if high_res else 1
            color = (0, 0, 0)  # Blue color

            thickness = 2 if high_res else 1  # Adjusted thickness based on resolution

           
            image_copy = image.copy()

            if twoline:
                image_with_text = cv2.putText(image_copy, IST.strftime("%T IST"), org, font, fontScale, color, thickness, cv2.LINE_AA)
                org = (org[0], org[1] + 25)
                image_with_text = cv2.putText(image_with_text, IST.strftime("%d-%m-%Y"), org, font, fontScale, color, thickness, cv2.LINE_AA)
            else:
                image_with_text = cv2.putText(image_copy, IST.strftime("%T IST %d-%m-%Y"), org, font, fontScale, color, thickness, cv2.LINE_AA)
            self.finalimage = cv2.cvtColor(image_with_text, cv2.COLOR_BGR2RGB)
            return 'Success'
        except Exception as e:
            return f"Error overlaying text on image: {e}"

    def process(self, filepath):
        filename = filepath.split('/')[-1]
        stn_name = filename.split('.')[0].split('_')[1]
        product_code = filename.split('.')[0].split('_')[0]

        #filter = self.configstn['stn_name'] == stn_name
        #filter2 = self.configstn['prodcut_code'] == product_code
        filter = self.configstn['stn_name'].str.lower() == stn_name.lower()
        filter2 = self.configstn['prodcut_code'].str.lower() == product_code.lower()
        parameters = self.configstn.loc[filter & filter2]
        try:
            parameters = parameters.iloc[0].to_dict()
            
        except:
            
            logmessasge = f'{self.timestamp}, {stn_name}, {product_code}, Station / product code not found in config.\n'
            self.writelog(self.errorlogfile, logmessasge)
            return logmessasge
        
        self.x_start = int(parameters['x_start'])
        self.y_start = int(parameters['y_start'])
        self.x_end = int(parameters['x_end'])
        self.y_end = int(parameters['y_end'])
        self.x1 = int(parameters['x1'])
        self.y1 = int(parameters['y1'])
        self.org = (self.x1, self.y1)
        self.apply_gaussian_blur = parameters['Gaussian Blur']
        self.twoline = parameters['Two Line']
        self.high_res = parameters['High Resolution']
        
        self.original_img, runstatus = self.readimage(filepath)
        if runstatus != 'Success':
            logmessasge = f'{self.timestamp}, {stn_name}, {product_code}, {runstatus}.\n'
            self.writelog(self.errorlogfile, logmessasge)
            status = 'Image read error'
            self.successlog(filepath,stn_name,product_code,parameters['Config_id'],np.nan,np.nan,status)
            return status

        
        runstatus = self.preprocess_image()
        if runstatus != 'Success':
            logmessasge = f'{self.timestamp}, {stn_name}, {product_code}, {runstatus}.\n'
            self.writelog(self.errorlogfile, logmessasge)
            status = 'Preprocess error'
            self.successlog(filepath,stn_name,product_code,parameters['Config_id'],np.nan,np.nan,status)
            return status
        
        runstatus = self.extract_time_text()
        if runstatus != 'Success':
            logmessasge = f'{self.timestamp}, {stn_name}, {product_code}, {runstatus}.\n'
            self.writelog(self.errorlogfile, logmessasge)
            status = 'OCR error'
            self.successlog(filepath,stn_name,product_code,parameters['Config_id'],self.preprocessedimg,np.nan,status)
            return status
        
        runstatus = self.convert_to_ist()
        if runstatus != 'Success':
            logmessasge = f'{self.timestamp}, {stn_name}, {product_code}, {runstatus}.\n'
            self.writelog(self.errorlogfile, logmessasge)
            status = 'IST conversion error'
            self.successlog(filepath,stn_name,product_code,parameters['Config_id'],self.preprocessedimg,self.convertedtxt,status)
            return status
        
        runstatus = self.overlay_text_on_image()
        if runstatus != 'Success':
            logmessasge = f'{self.timestamp}, {stn_name}, {product_code}, {runstatus}.\n'
            self.writelog(self.errorlogfile, logmessasge)
            status = 'Overlay error'
            self.successlog(filepath,stn_name,product_code,parameters['Config_id'],self.preprocessedimg,self.convertedtxt,status)
            return status
        
        imageio.mimsave(f'{self.op_path}/{filename}', [self.finalimage], duration=0.5)
        
        self.successlog(filepath,stn_name,product_code,parameters['Config_id'],self.preprocessedimg,self.convertedtxt,'Success')
        
        return "Success"
        
        #cv2.imwrite(f'{self.op_path}/{filename}', self.finalimage)

    def batchprocess(self):
        input_path = self.local_dir
        files = os.listdir(input_path)
        files = [f'{input_path}/{file}' for file in files if 'gif' in file]
        for file in files:
            status = self.process(file)
            print('Processing', file)
            if status != 'Success':
                print(f'Filed processing {file} - {status}')

    def viewlog(self, date, sttime = None, entime = None, StationID = None, ProductID=None, ConfigID = None):
        logfile = f'{self.log_path}/{date}.pickle'
        time = str(datetime.datetime.now()).replace(':','_').replace(' ', '_')
        export_path = f'{self.log_path}/{time}'
        os.makedirs(export_path, exist_ok=True)
        if os.path.exists(logfile):

            log = pd.read_pickle(logfile)
            #log.set_index('timestamp', inplace=True)
            if sttime is not None:
                if entime is not None:
                    sttime = f'{date} {sttime}'
                    entime = f'{date} {entime}'
                    mask = log['Timestamp'] >= sttime
                    mask1 = log['Timestamp'] <= entime
                    log = log[mask & mask1]
            if StationID is not None:
                mask = log['StationID'] == StationID
                log = log[mask]
            
            if ProductID is not None:
                mask = log['ProductID'] == ProductID
                log = log[mask]

            if ConfigID is not None:
                mask = log['ConfigID'] == ConfigID
                log = log[mask]
            image_data = log['Input_img']
            columns = [col for col in log.columns if col != 'Input_img']
            log = log[columns]
            #log['Timestamp'] = pd.to_datetime(log['Timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
            log.to_csv(f'{export_path}/log.csv')
            image_data = image_data.dropna()
            #print(image_data)
            for index, image in image_data.items():
                print(image, type(image))
                cv2.imwrite(f'{export_path}/{index}.jpg',image)
            #print(log,log.columns)

        else:
            print(f'{logfile} not found')

#utc = UtcConverter()
#utc.batchprocess() 
#utc.viewlog('2024-08-23')
if __name__=="__main__":
    #while True:
    utc = UtcConverter()
    #utc.download() 
    #utc.process('/home/irad2025/AGR/UTC2IST/Data/INPUT/caz_agt.gif')    
    st = datetime.datetime.now()
    utc.batchprocess()
    en = datetime.datetime.now()
    no_ip = len(os.listdir(utc.local_dir))
    no_op = len(os.listdir(utc.op_path))
    utc.viewlog('2024-08-30')
    #del utc
    
    print('Run Complete - ',str((en-st).total_seconds()/60),' minutes Time taken to run', f'{no_op} / {no_ip} has been processed successfully')
    #time.sleep(1800)
        #

