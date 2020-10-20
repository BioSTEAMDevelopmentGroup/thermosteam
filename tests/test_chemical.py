# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import pytest
from numpy.testing import assert_allclose

def test_chemical():
    import thermosteam as tmo
    CAS = ['12385-13-6', '7440-59-7', '7439-93-2', '7440-41-7', '7440-42-8', '7440-44    -0', '17778-88-0', '17778-80-2', '14762-94-8', '7440-01-9', '7440-23-5', '7439-95-4', '7429-90-5', '7440-21-3', '7723-14-0', '7704-34-9', '22537-15-1', '7440-37-1', '7440-09-7', '7440-70-2', '7440-20-2', '7440-32-6', '7440-62-2', '7440-47-3', '7439-96-5', '7439-89-6', '7440-48-4', '7440-02-0', '7440-50-8', '7440-66-6', '7440-55-3', '7440-56-4', '7440-38-2', '7782-49-2', '10097-32-2', '7439-90-9', '7440-17-7', '7440-24-6', '7440-65-5', '7440-67-7', '7440-03-1', '7439-98-7', '7440-26-8', '7440-18-8', '7440-16-6', '7440-05-3', '7440-22-4', '7440-43-9', '7440-74-6', '7440-31-5', '7440-36-0', '13494-80-9', '7553-56-2', '7440-63-3', '7440-46-2', '7440-39-3', '7439-91-0', '7440-45-1', '7440-10-0', '7440-00-8', '7440-12-2', '7440-19-9', '7440-53-1', '7440-54-2', '7440-27-9', '7429-91-6', '7440-60-0', '7440-52-0', '7440-30-4', '7440-64-4', '7439-94-3', '7440-58-6', '7440-25-7', '7440-33-7', '7440-15-5', '7440-04-2', '7439-88-5', '7440-06-4', '7440-57-5', '7439-97-6', '7440-28-0', '7439-92-1', '7440-69-9', '7440-08-6', '7440-68-8', '10043-92-2', '7440-73-5', '7440-14-4', '7440-34-8', '7440-29-1', '7440-13-3', '7440-61-1', '7439-99-8', '7440-07-5', '7440-35-9', '7440-51-9', '7440-40-6', '7440-71-3', '7429-92-7', '7440-72-4', '7440-11-1', '10028-14-5', '22537-19-5', '53850-36-5', '53850-35-4', '54038-81-2', '54037-14-8', '54037-57-9', '54038-01-6', '54083-77-1', '54386-24-2', '54084-26-3', '54084-70-7', '54085-16-4', '54085-64-2', '54100-71-9', '54101-14-3', '54144-19-3', '78-96-6', '97-00-7', '107-06-2', '120-82-1', '107-20-0', '107-07-3', '100-54-9', '541-50-4', '123-08-0', '107-21-1', '64-19-7', '75-07-0', '60-35-5', '513-86-0', '67-64-1', '124-04-9', '7664-41-7', '100-52-7', '71-43-2', '65-85-0', '100-51-6', '10035-10-6', '123-72-8', '513-85-9', '71-36-3', '107-92-6', '462-94-2', '124-38-9', '630-08-0', '64-18-6', '120-80-9', '74-82-8', '79-11-8', '77-92-9', '7647-01-0', '95-48-7', '69-72-7', '108-39-4', '111-65-9', '124-07-2', '111-16-0', '79-46-9', '7783-06-4', '109-76-2', '124-13-0', '90-05-1', '6915-15-7', '123-38-6', '132-64-9', '50-21-5', '431-03-8', '124-40-3', '67-68-5', '141-43-5', '64-17-5', '50-00-0', '75-12-7', '110-94-1', '56-40-6', '56-81-5', '141-46-8', '79-14-1', '74-90-8', '1333-74-0', '7722-84-1', '123-31-9', '7803-49-8', '288-32-4', '120-72-9', '97-65-4', '141-82-2', '133-37-9', '74-93-1', '67-56-1', '87-89-8', '103-84-4', '91-20-3', '59-67-6', '7697-37-2', '7727-37-9', '10024-97-2', '121-69-7', '111-87-5', '7732-18-5', '144-62-7', '7782-44-7', '57-10-3', '85-01-8', '108-95-2', '7664-38-2', '88-99-3', '57-55-6', '71-23-8', '79-09-4', '288-13-1', '110-86-1', '87-66-1', '127-17-3', '75-18-3', '110-15-6', '57-50-1', '7664-93-9', '7446-09-5', '110-01-0', '68-11-1', '108-88-3', '75-50-3', '57-13-6', '121-33-5', '51-28-5', '60-24-2', '148-24-3', '103-90-2', '498-02-2', '50-78-2', '123-99-9', '55-21-0', '50-32-8', '120-51-4', '57-57-8', '58-08-2', '76-22-2', '36653-82-4', '106-44-5', '1003-03-8', '334-48-5', '132-65-0', '74-95-3', '84-74-2', '1120-48-5', '119-61-9', '60-29-7', '107-15-3', '2809-21-4', '97-53-0', '111-30-8', '151-67-7', '15687-27-1', '67-63-0', '143-07-7', '60-33-3', '121-75-5', '79-41-4', '100-97-0', '119-36-8', '55-63-0', '106-51-4', '106-48-9', '106-46-7', '110-85-0', '7447-40-7', '7681-11-0', '108-46-3', '81-07-2', '111-20-6', '7647-14-5', '7681-49-4', '7681-82-5', '57-11-4', '111-48-8', '102-76-1', '75-25-2', '112-24-3', '50-70-4', '56-23-5', '56-55-3', '56-87-1', '5329-14-6', '57-88-5', '540-82-9', '58-72-0', '60-00-4', '60-09-3', '60-12-8', '104-15-4', '62-53-3', '62-75-9', '63-91-2', '64-67-5', '66-25-1', '67-66-3', '67-72-1', '68-12-2', '71-41-0', '71-55-6', '74-31-7', '74-83-9', '74-84-0', '74-85-1', '74-86-2', '74-87-3', '74-88-4', '74-89-5', '74-96-4', '74-97-5', '74-98-6', '74-99-7', '75-00-3', '75-01-4', '75-02-5', '75-03-6', '75-04-7', '75-05-8', '75-08-1', '75-09-2', '75-10-5', '75-11-6', '75-15-0', '75-19-4', '75-21-8', '7637-07-2', '75-26-3', '75-28-5', '75-29-6', '75-30-9', '75-31-0', '75-33-2', '75-34-3', '75-35-4', '75-36-5', '75-37-6', '75-38-7', '75-43-4', '75-44-5', '75-45-6', '75-46-7', '75-52-5', '75-55-8', '75-56-9', '75-61-6', '75-62-7', '75-63-8', '75-64-9', '75-65-0', '75-66-1', '75-68-3', '75-69-4', '75-71-8', '75-72-9', '75-73-0', '75-75-2', '75-76-3', '75-77-4', '75-78-5', '75-79-6', '75-83-2', '75-84-3', '75-85-4', '75-86-5', '75-87-6', '75-88-7', '75-91-2', '75-97-8', '75-98-9', '76-01-7', '76-02-8', '76-03-9', '76-05-1', '76-11-9', '76-12-0', '76-13-1', '76-14-2', '76-15-3', '76-16-4', '76-19-7', '77-47-4', '77-68-9', '77-73-6', '77-74-7', '77-78-1', '77-79-2', '77-99-6', '78-00-2', '107-96-0', '78-10-4', '78-11-5', '78-30-8', '78-40-0', '78-59-1', '78-75-1', '78-76-2', '78-78-4', '78-79-5', '78-81-9', '78-82-0', '78-83-1', '78-84-2', '78-85-3', '78-86-4', '78-87-5', '78-88-6', '78-90-0', '78-92-2', '78-93-3', '78-97-7', '78-99-9', '79-00-5', '79-01-6', '79-02-7', '79-04-9', '79-06-1', '79-10-7', '79-16-3', '79-20-9', '79-21-0', '79-22-1', '79-24-3', '79-27-6', '79-29-8', '79-31-2', '79-34-5', '79-36-7', '79-38-9', '79-39-0', '79-43-6', '79-92-5', '80-05-7', '80-10-4', '80-15-9', '80-43-3', '80-46-6', '80-47-7', '80-56-8', '80-62-6', '80-73-9', '83-32-9', '83-48-7', '84-15-1', '84-65-1', '84-66-2', '84-69-5', '84-75-3', '84-76-4', '84-77-5', '85-44-9', '86-73-7', '86-74-8', '87-41-2', '87-61-6', '87-68-3', '87-85-4', '88-09-5', '88-18-6', '88-20-0', '88-72-2', '88-73-3', '88-74-4', '88-85-7', '88-89-1', '89-05-4', '89-83-8', '89-95-2', '90-00-6', '90-02-8', '90-04-0', '90-11-9', '90-12-0', '90-13-1', '90-42-6', '91-08-7', '91-10-1', '493-01-6', '91-22-5', '91-23-6', '91-57-6', '91-63-4', '91-66-7', '92-06-8', '92-24-0', '92-51-3', '92-52-4', '92-67-1', '92-84-2', '92-87-5', '92-94-4', '93-51-6', '93-53-8', '93-54-9', '93-56-1', '93-58-3', '93-89-0', '94-28-0', '94-36-0', '95-13-6', '95-15-8', '95-47-6', '95-49-8', '95-50-1', '95-51-2', '95-53-4', '95-54-5', '95-57-8', '95-63-6', '95-65-8', '95-68-1', '95-73-8', '95-76-1', '95-80-7', '95-87-4', '95-92-1', '95-93-2', '95-96-5', '96-05-9', '96-10-6', '96-14-0', '96-17-3', '96-18-4', '96-22-0', '96-23-1', '96-24-2', '96-29-7', '96-31-1', '96-33-3', '96-34-4', '96-37-7', '96-47-9', '96-48-0', '96-49-1', '96-54-8', '97-62-1', '97-63-2', '97-64-3', '97-72-3', '97-85-8', '97-86-9', '97-88-1', '97-95-0', '97-99-4', '98-00-0', '98-01-1', '98-06-6', '98-07-7', '98-08-8', '98-11-3', '98-13-5', '98-29-3', '98-46-4', '98-54-4', '98-56-6', '98-66-8', '98-82-8', '98-83-9', '98-85-1', '98-86-2', '98-87-3', '98-88-4', '98-95-3', '99-04-7', '99-08-1', '99-09-2', '99-12-7', '99-35-4', '99-54-7', '99-62-7', '99-63-8', '99-65-0', '99-75-2', '99-85-4', '99-86-5', '99-87-6', '99-93-4', '99-94-5', '99-99-0', '100-00-5', '100-01-6', '100-10-7', '100-18-5', '100-20-9', '100-21-0', '100-25-4', '100-37-8', '100-40-3', '100-41-4', '100-42-5', '100-44-7', '100-46-9', '100-47-0', '100-50-5', '100-53-8', '100-60-7', '100-61-8', '100-63-0', '100-64-1', '100-66-3', '100-68-5', '100-74-3', '100-80-1', '101-54-2', '101-68-8', '101-81-5', '101-83-7', '101-84-8', '101-97-3', '102-01-2', '102-25-0', '102-36-3', '542-92-7', '102-69-2', '102-70-5', '102-71-6', '102-82-9', '10043-35-3', '103-09-3', '103-11-7', '103-23-1', '103-29-7', '103-50-4', '103-65-1', '103-69-5', '103-70-8', '103-71-9', '103-73-1', '103-76-4', '104-01-8', '104-46-1', '104-51-8', '104-57-4', '104-72-3', '104-76-7', '104-87-0', '105-05-5', '105-08-8', '105-30-6', '105-34-0', '105-37-3', '105-38-4', '105-39-5', '105-45-3', '105-46-4', '105-53-3', '105-54-4', '105-56-6', '105-57-7', '105-58-8', '105-59-9', '105-60-2', '105-66-8', '105-67-9', '106-20-7', '106-27-4', '106-31-0', '106-32-1', '106-33-2', '106-35-4', '106-36-5', '106-37-6', '106-38-7', '106-42-3', '106-43-4', '106-47-8', '106-49-0', '106-50-3', '106-63-8', '106-65-0', '106-88-7', '106-89-8', '106-92-3', '106-93-4', '106-94-5', '106-97-8', '106-98-9', '106-99-0', '107-00-6', '107-02-8', '107-03-9', '107-05-1', '107-10-8', '107-11-9', '107-12-0', '107-13-1', '107-16-4', '107-18-6', '107-19-7', '107-22-2', '107-25-5', '107-29-9', '107-30-2', '107-31-3', '107-39-1', '107-40-4', '107-41-5', '107-47-1', '107-52-8', '107-83-5', '107-87-9', '107-88-0', '107-89-1', '107-98-2', '108-01-0', '108-03-2', '108-05-4', '108-08-7', '108-10-1', '108-11-2', '108-18-9', '108-20-3', '108-21-4', '108-22-5', '108-24-7', '108-29-2', '108-30-5', '108-31-6', '108-32-7', '108-36-1', '108-38-3', '108-42-9', '108-43-0', '108-44-1', '108-45-2', '108-48-5', '108-55-4', '108-57-6', '108-59-8', '108-64-5', '108-65-6', '108-67-8', '108-68-9', '108-70-3', '108-75-8', '108-78-1', '108-82-7', '108-83-8', '108-86-1', '108-87-2', '108-89-4', '108-90-7', '108-91-8', '108-93-0', '108-94-1', '108-98-5', '108-99-6', '109-06-8', '109-21-7', '109-43-3', '109-49-9', '109-52-4', '109-55-7', '109-60-4', '109-64-8', '109-65-9', '109-66-0', '109-67-1', '109-69-3', '109-73-9', '109-74-0', '109-75-1', '109-77-3', '109-78-4', '109-79-5', '109-83-1', '109-86-4', '109-87-5', '109-89-7', '109-92-2', '109-93-3', '109-94-4', '109-97-7', '109-99-9', '110-00-9', '110-02-1', '110-05-4', '110-12-3', '110-18-9', '110-19-0', '110-27-0', '110-33-8', '110-38-3', '110-42-9', '110-43-0', '110-54-3', '110-56-5', '110-58-7', '110-59-8', '110-61-2', '110-62-3', '110-63-4', '110-65-6', '110-66-7', '110-71-4', '110-74-7', '110-77-0', '110-80-5', '110-81-6', '110-82-7', '110-83-8', '110-88-3', '110-89-4', '110-91-8', '110-96-3', '110-97-4', '110-99-6', '111-01-3', '111-11-5', '111-13-7', '111-14-8', '111-15-9', '111-26-2', '111-27-3', '111-29-5', '111-31-9', '111-34-2', '111-36-4', '111-40-0', '111-41-1', '111-42-2', '111-43-3', '111-44-4', '111-46-6', '111-47-7', '111-49-9', '111-55-7', '111-61-5', '111-62-6', '111-66-0', '111-68-2', '111-69-3', '111-70-6', '111-71-7', '111-76-2', '111-77-3', '111-78-4', '111-82-0', '111-84-2', '111-86-4', '111-88-6', '111-90-0', '111-91-1', '111-92-2', '111-96-6', '112-05-0', '112-06-1', '112-07-2', '112-14-1', '112-15-2', '112-17-4', '112-23-2', '112-25-4', '112-27-6', '112-30-1', '112-31-2', '112-32-3', '112-34-5', '112-35-6', '112-36-7', '112-37-8', '112-39-0', '112-40-3', '112-41-4', '112-42-5', '112-44-7', '112-49-2', '112-50-5', '112-53-8', '112-54-9', '112-55-0', '112-57-2', '112-58-3', '112-59-4', '112-60-7', '112-61-8', '112-63-0', '112-70-9', '112-72-1', '112-73-2', '112-86-7', '112-88-9', '112-92-5', '112-95-8', '115-07-1', '115-10-6', '115-11-7', '115-21-9', '115-25-3', '115-77-5', '115-86-6', '116-09-6', '116-14-3', '116-15-4', '116-53-0', '117-81-7', '118-71-8', '118-74-1', '118-90-1', '118-91-2', '118-93-4', '118-96-7', '119-64-2', '119-65-3', '119-67-5', '119-75-5', '120-12-7', '120-61-6', '120-92-3', '120-94-5', '121-14-2', '121-17-5', '121-32-4', '121-34-6', '121-43-7', '121-44-8', '121-57-3', '121-73-3', '121-82-4', '121-91-5', '131-11-3', '131-16-8', '131-17-9', '134-96-3', '135-01-3', '135-98-8', '136-35-6', '149-57-5', '136-60-7', '137-32-6', '138-87-4', '139-87-7', '140-11-4', '140-29-4', '140-31-8', '140-66-9', '140-88-5', '141-32-2', '141-59-3', '141-62-8', '141-63-9', '141-78-6', '141-79-7', '141-93-5', '141-97-9', '142-28-9', '142-29-0', '142-62-1', '142-68-7', '142-82-5', '142-84-7', '142-91-6', '142-92-7', '142-96-1', '143-08-8', '2016-57-1', '143-10-2', '143-13-5', '143-15-7', '143-22-6', '143-24-8', '143-33-9', '144-19-4', '149-74-6', '150-76-5', '151-56-4', '156-43-4', '156-87-6', '205-99-2', '206-44-0', '208-96-8', '217-59-4', '218-01-9', '260-94-6', '279-23-2', '280-33-1', '280-57-9', '281-23-2', '287-23-0', '287-27-4', '287-92-3', '288-14-2', '288-42-6', '289-80-5', '289-95-2', '290-37-9', '291-64-5', '292-64-8', '301-00-8', '302-01-2', '306-83-2', '307-34-6', '320-60-5', '335-57-9', '352-93-2', '353-36-6', '353-50-4', '353-59-3', '354-23-4', '354-33-6', '354-58-5', '355-25-9', '355-42-0', '356-18-3', '359-10-4', '359-35-3', '367-11-3', '372-18-9', '374-07-2', '375-22-4', '392-56-3', '409-21-2', '420-12-2', '420-46-2', '430-66-0', '460-19-5', '461-58-5', '462-06-6', '462-95-3', '463-49-0', '463-51-4', '463-58-1', '463-82-1', '464-06-2', '471-34-1', '127-27-5', '471-77-2', '479-45-8', '488-23-3', '496-11-7', '496-14-0', '497-04-1', '497-19-8', '497-26-7', '498-66-8', '501-65-5', '502-44-3', '502-56-7', '503-17-3', '503-30-0', '503-74-2', '504-63-2', '505-22-6', '505-48-6', '506-12-7', '506-30-9', '506-77-4', '507-20-0', '509-14-8', '512-56-1', '513-35-9', '513-36-0', '513-44-0', '513-53-1', '513-77-9', '513-81-5', '514-10-3', '519-73-3', '526-73-8', '526-75-0', '527-53-7', '527-84-4', '528-29-0', '528-44-9', '529-20-4', '534-15-6', '535-77-3', '536-50-5', '536-74-3', '538-23-8', '538-24-9', '538-68-1', '538-93-2', '539-30-0', '539-88-8', '540-18-1', '540-36-3', '540-54-5', '540-61-4', '540-63-6', '540-67-0', '540-84-1', '540-88-5', '540-97-6', '541-01-5', '541-02-6', '541-05-9', '541-41-3', '541-73-1', '542-18-7', '542-55-2', '542-69-8', '542-88-1', '543-49-7', '543-59-9', '544-02-5', '544-13-8', '544-35-4', '544-40-1', '544-63-8', '544-76-3', '544-85-4', '547-63-7', '547-64-8', '552-30-7', '554-12-1', '554-14-3', '555-10-2', '555-43-1', '555-44-2', '555-45-3', '556-52-5', '556-67-2', '556-68-3', '556-69-4', '557-17-5', '557-40-4', '557-91-5', '557-98-2', '558-30-5', '558-37-2', '560-21-4', '562-49-2', '563-16-6', '563-43-9', '563-45-1', '563-46-2', '563-78-0', '563-79-1', '563-80-4', '564-02-3', '565-59-3', '565-61-7', '565-69-5', '565-75-3', '565-80-0', '573-56-8', '576-26-1', '578-54-1', '579-66-8', '581-42-0', '582-16-1', '583-48-2', '584-02-1', '584-03-2', '584-08-7', '123-56-8', '584-84-9', '584-94-1', '585-07-9', '585-34-2', '586-62-9', '587-03-1', '122-39-4', '103-30-0', '589-18-4', '589-34-4', '589-35-5', '589-38-8', '589-43-5', '589-53-7', '589-63-9', '589-81-1', '589-82-2', '2207-04-7', '590-01-2', '590-19-2', '590-35-2', '590-66-9', '590-67-0', '590-73-8', '590-86-3', '591-50-4', '123-76-2', '591-76-4', '591-78-6', '591-87-7', '591-93-5', '591-95-7', '592-13-2', '592-27-8', '592-41-6', '592-42-7', '592-45-0', '13269-52-8', '592-57-4', '592-76-7', '6443-92-1', '14686-14-7', '592-84-7', '592-88-1', '14850-23-8', '593-45-3', '593-49-7', '593-53-3', '593-60-2', '593-70-4', '593-74-8', '594-11-6', '594-44-5', '594-56-9', '594-61-6', '594-82-1', '595-37-9', '598-03-8', '598-25-4', '598-50-5', '598-53-8', '598-73-2', '598-75-4', '599-64-4', '603-35-0', '604-88-6', '605-01-6', '605-02-7', '124-68-5', '606-20-2', '609-26-7', '610-39-9', '611-14-3', '611-15-4', '611-32-5', '612-00-0', '616-02-4', '616-21-7', '616-23-9', '616-38-6', '616-39-7', '616-44-4', '616-45-5', '617-78-7', '617-94-7', '618-85-9', '619-15-8', '619-66-9', '619-99-8', '620-02-0', '620-14-4', '620-17-7', '620-23-5', '621-77-2', '622-45-7', '622-96-8', '622-97-9', '623-27-8', '623-37-0', '623-42-7', '623-81-4', '141-05-9', '624-42-0', '624-48-6', '624-65-7', '624-72-6', '624-83-9', '624-89-5', '624-92-0', '625-22-9', '625-27-4', '625-44-5', '625-45-6', '625-54-7', '625-69-4', '625-80-9', '626-67-5', '626-93-7', '627-05-4', '627-19-0', '627-21-4', '627-30-5', '627-58-7', '627-98-5', '628-02-4', '628-28-4', '628-29-5', '628-32-0', '628-41-1', '628-55-7', '628-63-7', '628-71-7', '628-73-9', '628-76-2', '628-81-9', '628-92-2', '628-97-7', '628-99-9', '629-04-9', '629-05-0', '629-11-8', '629-14-1', '629-19-6', '629-20-9', '629-45-8', '629-50-5', '629-59-4', '629-62-9', '629-73-2', '629-76-5', '629-78-7', '629-82-3', '629-92-5', '629-94-7', '629-96-9', '629-97-0', '629-99-2', '630-01-3', '630-02-4', '630-03-5', '630-06-8', '630-20-6', '630-76-2', '631-36-7', '873-66-5', '637-92-3', '638-02-8', '638-45-9', '638-49-3', '638-53-9', '638-67-5', '638-68-6', '644-49-5', '645-62-5', '627-20-3', '646-30-0', '646-31-1', '652-67-5', '659-70-1', '661-19-8', '674-82-8', '675-62-7', '706-31-0', '677-21-4', '678-26-2', '680-31-9', '681-84-5', '684-16-2', '688-74-4', '689-12-3', '689-97-4', '690-39-1', '691-37-2', '693-02-7', '693-23-2', '693-65-2', '693-89-0', '695-12-5', '696-29-7', '700-12-9', '702-79-4', '717-74-8', '760-20-3', '760-21-4', '760-23-6', '763-29-1', '763-69-9', '764-13-6', '928-53-0', '764-93-2', '767-58-8', '767-59-9', '771-61-9', '778-22-3', '791-28-6', '811-97-2', '812-04-4', '814-78-8', '818-61-1', '821-38-5', '821-55-6', '821-95-4', '822-06-0', '822-50-4', '823-40-5', '823-76-7', '827-52-1', '828-00-2', '832-64-4', '836-30-6', '868-77-9', '871-83-0', '872-05-9', '872-50-4', '872-55-9', '874-35-1', '874-41-9', '877-44-1', '882-33-7', '917-92-0', '919-30-2', '919-31-3', '919-94-8', '925-60-0', '10075-38-4', '927-49-1', '927-62-8', '928-49-4', '929-06-6', '124-22-1', '930-68-7', '933-98-2', '934-74-7', '934-80-5', '939-27-5', '959-26-2', '999-55-3', '999-97-3', '1002-43-3', '1002-84-2', '1003-38-9', '1012-72-2', '1066-33-7', '1067-08-9', '1067-20-5', '1067-53-4', '1068-87-7', '1070-87-7', '1071-26-7', '1071-81-4', '1072-05-5', '1072-16-8', '1074-17-5', '1074-43-7', '1074-55-1', '619-82-9', '1077-16-3', '1078-71-3', '1081-77-2', '1113-38-8', '1115-20-4', '1116-54-7', '1116-76-3', '1119-40-0', '1119-85-3', '1120-21-4', '1120-28-1', '1120-36-1', '1120-62-3', '1123-85-9', '1127-76-0', '1141-38-4', '1166-18-3', '1185-39-3', '1186-53-4', '1187-93-5', '1191-25-9', '1192-18-3', '1344-28-1', '1305-78-8', '1309-48-4', '1310-58-3', '1310-73-2', '1313-60-6', '1314-13-2', '16752-60-6', '1314-98-3', '1589-47-5', '127-91-3', '1333-82-0', '7664-39-3', '1336-21-6', '1345-25-1', '59-02-9', '1445-79-0', '1454-85-9', '1455-21-6', '1459-09-2', '1459-10-5', '1459-93-4', '1460-02-2', '1476-11-5', '1551-21-9', '1559-35-9', '1559-81-5', '1560-96-9', '1560-97-0', '1569-01-3', '1569-02-4', '1569-69-3', '1571-08-0', '1574-41-0', '1632-16-2', '1634-04-4', '1634-09-9', '1638-26-2', '1639-09-4', '1640-89-7', '1647-16-1', '1656-48-0', '1678-91-7', '1678-92-8', '1678-93-9', '1678-98-4', '1679-51-2', '1679-64-7', '1694-31-1', '1708-29-8', '1717-00-6', '1719-53-5', '124-18-5', '1740-19-8', '1741-83-9', '1746-23-2', '1757-42-2', '1758-88-9', '1759-53-1', '1759-58-6', '1759-81-5', '1779-25-5', '1795-09-1', '1795-16-0', '1814-88-6', '1945-53-5', '1948-33-0', '1962-75-0', '112-20-9', '2016-42-4', '2040-95-1', '2040-96-2', '2043-61-0', '2050-92-2', '2051-30-1', '124-09-4', '2131-18-2', '2177-47-1', '2189-60-8', '2207-01-4', '2210-28-8', '2216-33-3', '2216-34-4', '2216-51-5', '2314-97-8', '2315-68-6', '2425-74-3', '2432-74-8', '2437-56-1', '98-55-5', '2459-10-1', '2495-27-4', '2530-83-8', '2551-62-4', '2687-91-4', '2691-41-0', '2696-92-6', '2807-30-9', '2837-89-0', '2870-04-4', '2935-90-2', '3031-73-0', '3048-64-4', '3068-00-6', '3073-66-3', '3173-53-3', '3173-72-6', '3178-22-1', '3221-61-2', '3228-02-2', '3268-49-3', '3319-31-1', '3404-61-3', '3452-07-1', '3452-09-3', '3454-07-7', '3522-94-9', '3648-20-2', '3648-21-3', '3710-84-7', '503-64-0', '3769-23-1', '3875-51-2', '3877-15-4', '3913-02-8', '3938-95-2', '3944-36-3', '3982-91-0', '4038-04-4', '4048-33-3', '4050-45-7', '4067-16-7', '4110-50-3', '123-96-6', '15798-64-8', '4265-25-2', '4292-92-6', '4351-54-6', '4390-04-9', '4394-85-8', '4420-74-0', '4435-50-1', '4516-69-2', '4536-23-6', '4553-62-2', '5131-66-8', '5835-26-7', '5878-19-3', '5911-04-6', '6012-97-1', '6032-29-7', '6094-02-6', '6163-66-2', '6422-86-2', '6484-52-2', '6742-54-7', '6765-39-5', '6795-87-5', '6834-92-0', '6846-50-0', '6863-58-7', '6975-98-0', '7045-71-8', '7058-01-7', '7133-46-2', '7146-60-3', '7154-80-5', '7446-70-0', '7447-39-4', '7487-88-9', '7487-94-7', '7525-62-4', '7558-79-4', '7564-63-8', '7581-97-7', '7601-54-9', '7601-90-3', '7616-94-6', '7631-99-4', '7646-78-8', '7647-19-0', '7790-92-3', '7705-08-0', '7719-09-7', '7719-12-2', '7720-78-7', '7722-76-1', '7727-18-6', '7733-02-0', '7757-82-6', '7758-11-4', '7758-94-3', '7758-98-7', '7772-98-7', '7775-14-6', '7778-18-9', '7778-54-3', '7778-85-0', '7782-39-0', '7782-41-4', '7782-50-5', '7782-77-6', '7782-92-5', '7783-28-0', '7783-35-9', '7783-54-2', '7783-61-1', '7784-34-1', '7786-29-0', '7786-81-4', '7789-20-0', '7789-21-1', '7790-91-2', '7790-94-5', '7790-98-9', '7791-25-5', '7803-63-6', '7446-11-9', '107-51-7', '10061-02-6', '29911-28-2', '107-46-0', '10025-87-3', '10025-91-9', '10026-04-7', '10026-13-8', '10028-15-6', '10034-85-2', '10036-47-2', '10049-04-4', '10049-60-2', '13952-84-6', '10124-56-8', '10196-04-0', '10294-33-4', '10294-34-5', '10377-60-3', '10486-19-8', '10544-72-6', '10545-99-0', '10588-01-9', '12124-99-1', '12125-02-9', '13048-33-4', '13360-61-7', '13450-90-3', '13463-40-6', '13463-67-7', '13475-81-5', '3074-71-3', '15869-85-9', '15870-10-7', '15890-40-1', '16219-75-3', '16747-30-1', '16747-32-3', '16747-38-9', '16747-50-5', '17301-94-9', '17312-57-1', '17851-27-3', '18435-45-5', '18835-33-1', '19089-47-5', '7372-88-5', '122-32-7', '122-51-0', '122-52-1', '122-60-1', '122-66-7', '122-79-2', '122-97-4', '123-01-3', '123-02-4', '123-05-7', '123-07-9', '123-19-3', '123-25-1', '123-39-7', '123-42-2', '123-51-3', '123-54-6', '123-62-6', '123-63-7', '123-75-1', '123-79-5', '123-86-4', '123-91-1', '123-92-2', '123-95-5', '124-02-7', '124-06-1', '124-10-7', '124-11-8', '124-17-4', '124-19-6', '124-63-0', '124-70-9', '124-73-2', '126-13-6', '126-30-7', '126-33-0', '126-73-8', '126-98-7', '126-99-8', '127-00-4', '127-18-4', '127-19-5', '128-37-0', '129-00-0', '24800-44-0', '608-23-1', '615-60-1', '20348-51-0', '14850-22-7', '111-03-5', '56-86-0', '1008-17-9', '26761-40-0', '764-35-2', '107-08-4', '6434-78-2', '2765-18-6', '27554-26-3', '2958-75-0', '632-16-6', '30453-31-7', '4259-00-1', '32970-45-9', '38842-05-6', '112-47-0', '42205-08-3', '50623-57-9', '628-96-6', '22663-61-2', '15869-80-4', '98-49-7', '110-67-8', '115-84-4', '123-18-2', '149-31-5', '354-56-3', '591-68-4', '622-40-2', '625-60-5', '629-33-4', '762-75-4', '840-65-3', '869-29-4', '925-78-0', '994-05-8', '999-61-1', '1120-34-9', '10025-85-1', '10544-73-7', '1446-61-3', '78-80-8', '93-96-9', '1511-62-2', '2084-19-7', '2530-87-2', '2628-17-3', '2897-60-1', '3031-74-1', '4654-26-6', '5332-52-5', '7758-89-6', '7783-55-3', '624-64-6', '105-06-6', '110-06-5', '1066-40-6', '1112-39-6', '1115-08-8', '1115-99-7', '10038-98-9', '10102-03-1', '10147-36-1', '10377-51-2', '10522-26-6', '96-35-5', '26519-91-5', '122-96-3', '354-38-1', '355-49-7', '373-04-6', '375-85-9', '406-58-6', '431-89-0', '460-73-1', '558-43-0', '594-39-8', '598-04-9', '598-23-2', '618-46-2', '621-71-6', '627-02-1', '628-80-8', '628-87-5', '632-51-9', '662-01-1', '679-86-7', '685-63-2', '765-30-0', '766-07-4', '837-08-1', '993-10-2', '15507-13-8', '1191-87-3', '1191-99-7', '1195-14-8', '1123-00-8', '3228-03-3', '1528-48-9', '1551-27-5', '1559-34-8', '1453-24-3', '497-25-6', '1599-67-3', '1691-17-4', '1795-05-7', '1805-22-7', '1885-48-9', '1889-67-4', '2039-93-2', '2110-78-3', '2150-02-9', '2163-42-0', '2274-11-5', '2366-36-1', '2456-27-1', '2550-06-3', '2652-13-3', '2690-08-6', '2752-17-2', '2768-02-7', '999-21-3', '3006-96-0', '3010-96-6', '3238-40-2', '3385-78-2', '3524-73-0', '3645-00-9', '3682-91-5', '3698-94-0', '3822-68-2', '4062-60-6', '4253-34-3', '4439-20-7', '4445-07-2', '4485-09-0', '4588-18-5', '307-59-5', '431-63-0', '537-40-6', '589-40-2', '626-95-9', '632-15-5', '872-93-5', '1184-58-3', '1191-41-9', '5451-52-5', '5500-21-0', '5650-20-4', '6145-31-9', '1454-84-8', '6228-73-5', '6294-34-4', '6362-80-7', '6881-94-3', '628-08-0', '7209-38-3', '7307-55-3', '7327-60-8', '7351-61-3', '7364-19-4', '7782-78-7', '10192-32-2', '10220-23-2', '10493-43-3', '10605-40-0', '13027-88-8', '13465-77-5', '13831-30-6', '2432-63-5', '14752-75-1', '14814-09-6', '15869-87-1', '16587-40-9', '15403-89-1', '16630-91-4', '16746-86-4', '138495-42-8', '17312-44-6', '17689-77-9', '18328-90-0', '18435-53-5', '18835-34-2', '20207-13-0', '21282-97-3', '23778-52-1', '23783-42-8', '24615-84-7', '24800-25-7', '4454-05-1', '2998-08-5', '3071-32-7', '13151-34-3', '56706-10-6', '57018-52-7', '7154-79-2', '4032-94-4', '25714-71-0', '698-87-3', '3004-93-1', '3726-45-2', '25961-89-1', '37910-77-3', '6064-63-7', '1502-22-3', '925-54-2', '3404-58-8', '3525-25-5', '6418-41-3', '6047-69-4', '5756-43-4', '34885-03-5', '33021-02-2', '307-45-9', '29911-27-1', '2292-79-7', '2752-99-0', '4813-50-7', '7789-25-5', '993-07-7', '20333-39-5', '37143-54-7', '371-78-8', '373-80-8', '591-96-8', '592-44-9', '598-26-5', '632-50-8', '646-05-9', '694-92-8', '1520-42-9', '1860-27-1', '2980-71-4', '3055-14-9', '3878-46-4', '6163-64-0', '6737-11-7', '10410-35-2', '13511-13-2', '13556-58-6', '14290-92-7', '15869-89-3', '10025-78-2', '19269-28-4', '10102-43-9', '10604-69-0', '4098-71-9', '2876-53-1', '54839-24-6', '1323-65-5', '17496-08-1', '821-11-4', '55334-42-4', '132259-10-0', '83-46-5', '692-45-5', '542-10-9', '60435-70-3', '5451-92-3', '5454-79-5', '1821-27-8', '67-47-0', '21956-56-9', '106-68-3', '766-90-5', '7758-02-3', '7647-15-6', '3377-92-2', '7775-38-4', '4538-37-8', '2038-03-1', '115-09-3', '5470-11-1', '110-16-7', '140-10-3', '110-17-8', '112-80-1', '3603-45-0', '144-55-8', '7775-09-9', '7681-38-1', '127-08-2', '127-09-3', '532-32-1', '631-61-8', '13151-05-8', '13151-06-9', '13463-39-3', '1551-32-2', '17312-77-5', '18970-30-4', '1961-96-2', '25360-10-5', '26158-99-6', '2884-06-2', '3404-71-5', '375-96-2', '38514-05-5', '6117-98-2', '6196-58-3', '871-28-3', '13286-92-5', '18912-81-7', '17059-44-8', '148-18-5', '21460-36-6', '10441-57-3', '63444-56-4', '4286-23-1', '4773-83-5', '40650-41-7', '28553-12-0', '107-93-7', '764-42-1', '931-88-4', '14919-01-8', '4923-91-5', '110-57-6', '7688-21-3', '149-30-4', '1207-12-1', '62-56-6', '141-53-7', '498-07-7', '4427-96-7', '375-03-1', '22410-44-2', '1964-45-0', '38514-03-3', '59919-41-4', '65185-89-9', '546-89-4', '35112-74-4', '51526-06-8', '1100-10-3', '463-40-1', '590-18-1', '623-36-9', '646-04-8', '645-49-8', '6434-77-1', '112-62-9', '14465-68-0', '1305-62-0', '7783-20-2', '1590-87-0', '1066-35-9', '1111-74-6', '13465-78-6', '4109-96-0', '75-54-7', '2487-90-3', '19287-45-7', '992-94-9', '3811-04-9', '55735-23-4', '14762-55-1', '7331-52-4', '142-47-2', '51655-57-3', '21482-12-2', '13349-10-5', '21078-81-9', '10192-30-0', '66325-11-9', '2141-58-4', '124-41-4', '359-15-9', '148462-57-1', '38433-80-6', '23305-64-8', '64001-06-5', '94023-15-1', '31283-14-4', '19177-04-9', '78-01-3', '39972-78-6', '61868-14-2', '78448-33-6', '75-24-1', '97-93-8', '100-99-2', '2471-08-1', '78024-33-6', '5809-59-6', '1284-72-6', '16940-66-2', '7681-52-9', '7631-90-5', '7632-00-0', '13453-80-0', '7558-80-7', '577-11-7', '19295-81-9', '50-81-7']
    for i in CAS: tmo.Chemical(i) # Simply make sure they can be created without errors
    

if __name__ == '__main__':
    test_chemical()