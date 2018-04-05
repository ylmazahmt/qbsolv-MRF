from subprocess import call
import sys
import os


def main():
	
	file_name = sys.argv[1]
	img_name, ext = file_name.split(".")
	rel_path = "result/" +img_name + "/"
	img_source_path = "../../img/" + file_name
	
	if(os.path.isfile(img_source_path) != 1):
		print("File (",img_source_path,") does not exist!")
		return
	
	if(os.path.isdir("result/" + img_name) != 1):
		call(["mkdir","result/" + img_name])
	
	output_file_name = str(img_name) + "_out." + str(ext)
	seg_file_name = img_name + "_segmentation.qubo"
	seg_out_file_name = img_name + "_segmentation.qbout"
	
	print("-->MRF Model being created...")
	call(["python3", "qubo_creator.py", img_source_path])
	print("\tSuccesfull!",seg_file_name)
	
	print("-->Qbsolv running on created model...")
	call(["../../qbsolv-master/build/qbsolv", "-i", rel_path + seg_file_name, "-o", rel_path + seg_out_file_name, "-v1", "-n2"])
	print("\tSuccesfull!",seg_out_file_name)
	
	print("-->Resulting image is writing out as image file...")
	call(["python3", "image_output.py", img_source_path])
	print("\tSuccesfull",output_file_name )
	
	print("ALL DONE!")
	
	if(os.path.isfile(rel_path + output_file_name)):
		call(["open", rel_path + output_file_name])
		return

if __name__=="__main__":
	main()

