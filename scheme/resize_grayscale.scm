(define (batch-resize)
	(let* ((filelist (cadr (file-glob "*.jpeg" 1))))
		(while (not (null? filelist))
            (let* (
                    (filename (car filelist))
                    (image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
                    (drawable (car (gimp-image-active-drawable image)))
					(width 256)
					(height 256)
                )
                (gimp-message filename)
                (gimp-image-scale-full image width height INTERPOLATION-CUBIC)
				; (gimp-image-convert-grayscale image)
                (let 
                    ((nfilename (string-append "thumb_" filename)))
                    (gimp-file-save RUN-NONINTERACTIVE image drawable nfilename nfilename)
                )
                (gimp-image-delete image)
            )
            (set! filelist (cdr filelist))
        )
	)
)
(script-fu-register "batch-resize"
	""
	"Do nothing"
	"Joey User"
	"Joey User"
	"August 2000"
	""
)