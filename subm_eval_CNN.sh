#!/bin/sh
#
#(make sure the right shell will be used)
#$ -S /bin/sh
#
#(the cpu time for this job)
#$ -l h_cpu=1:29:00
#
#(the maximum memory usage of this job)
#$ -l h_vmem=5000M
#
#(use hh site)
#$ -l site=hh
#(stderr and stdout are merged together to stdout)
#$ -j y
#
# use SL5
#$ -l os=sld6
#
# use current dir and current environment
#$ -cwd
#$ -V
#

config=$2
cd /nfs/dust/cms/user/bukinkir/GeomTauId/
while read line
do
cd /nfs/dust/cms/user/bukinkir/GeomTauId/
cd jobs
lt=`echo $line`
_lt=`echo $line | cut -d '/' -f2`
		cat bss > job_${_lt}.sh
		echo python /nfs/dust/cms/user/bukinkir/GeomTauId/EvalModel.py --config=$config --file=$lt >> job_${_lt}.sh
		chmod 777 job_${_lt}.sh
		chmod +x job_${_lt}.sh
		./HTC_submit.sh job_${_lt}.sh ${_lt}


done<$1
