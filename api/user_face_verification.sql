CREATE DATABASE User_Face_verification_system;

USE User_Face_verification_system;

CREATE TABLE UserAdmin (
    UserID INT PRIMARY KEY IDENTITY(1,1),
    Name NVARCHAR(60) NOT NULL,         
    Major NVARCHAR(20) NOT NULL,        
    IDNumber NVARCHAR(11) NOT NULL,      
    Email NVARCHAR(60) NOT NULL,        
    Address NVARCHAR(150),               
    Phone NVARCHAR(12),                  
    LinkedIn NVARCHAR(100),
	IMG_SCR NVARCHAR(50));

CREATE TABLE UserStudent (
    UserID INT PRIMARY KEY IDENTITY(1,1),
    Name NVARCHAR(60) NOT NULL,         
    Major NVARCHAR(20) NOT NULL,        
    IDNumber NVARCHAR(11) NOT NULL,      
    Email NVARCHAR(60) NOT NULL,        
    Address NVARCHAR(150),               
    Phone NVARCHAR(12),                  
    LinkedIn NVARCHAR(100),
	IMG_SCR NVARCHAR(50));

CREATE TABLE UserVisiter (
    UserID INT PRIMARY KEY IDENTITY(1,1),
    Name NVARCHAR(60) NOT NULL,         
    Major NVARCHAR(20) NOT NULL,        
    IDNumber NVARCHAR(11) NOT NULL,      
    Email NVARCHAR(60) NOT NULL,        
    Address NVARCHAR(150),               
    Phone NVARCHAR(12),                  
    LinkedIn NVARCHAR(100),
	IMG_SCR NVARCHAR(50));

--INSERT INTO UserAdmin (Name, Major, IDNumber, Email, Address, Phone, LinkedIn)
--VALUES 
--(
--    'John Doe', 
--    'Computer Science', 
--    'CS123456789', 
--    'johndoe@example.com', 
--    '123 Elm Street, Springfield, IL', 
--    '9876543210', 
--    'https://www.linkedin.com/in/johndoe'
--);

--DELETE FROM UserAdmin WHERE Name = 'John Doe';
--DELETE FROM UserAdmin;
--DELETE FROM UserStudent;
--DELETE FROM UserVisiter;

SELECT * FROM UserAdmin;
SELECT * FROM UserStudent;
SELECT * FROM UserVisiter;