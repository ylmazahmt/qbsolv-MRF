from subprocess import call
import sys


def main():
	file_name = sys.argv[1]
	img_name, ext = file_name.split(".")
	output_file_name = str(img_name) + "_out." + str(ext)
	seg_file_name = img_name + "_segmentation.qubo"
	seg_out_file_name = img_name + "_segmentation.qbout"
	print("-->MRF Model being created...")
	call(["python3", "qubo_creator.py", file_name])
	print("\tSuccesfull!",seg_file_name)
	print("-->Qbsolv running on created model...")
	call(["../../qbsolv-master/build/qbsolv", "-i", seg_file_name, "-o", seg_out_file_name, "-v1", "-n2"])
	print("\tSuccesfull!",seg_out_file_name)
	print("-->Resulting image is writing out as image file...")
	call(["python3", "image_output.py", file_name])
	print("\tSuccesfull",output_file_name )
	print("ALL DONE!")

if __name__=="__main__":
	main()

