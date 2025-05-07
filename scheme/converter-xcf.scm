(define (converter-xcf)
	(let* ((filelist (cadr (file-glob "*.jpeg" 1))))
		(while (not (null? filelist))
            (let* (
                    (filename (car filelist))
                    (only_filename (substring filename 0 (- (string-length filename) 5)))
                    (image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
                    (drawable (car (gimp-image-active-drawable image)))
                )
                (gimp-message filename)
                (let 
                    ((nfilename (string-append only_filename ".xcf")))
                    (gimp-xcf-save RUN-NONINTERACTIVE image drawable nfilename nfilename)
                )
                (gimp-image-delete image)
            )
            (set! filelist (cdr filelist))
        )
	)
)
(script-fu-register "converter-xcf"
	""
	"Do nothing"
	"Joey User"
	"Joey User"
	"August 2000"
	""
)