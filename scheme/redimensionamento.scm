(define (redimensionamento value)
	(let* ((filelist (cadr (file-glob "*.xcf" 1))))
		(while (not (null? filelist))
            (let* (
                    (filename (car filelist))
                    (only_filename (substring filename 0 (- (string-length filename) 4)))
                    (image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
                    (layer (car (gimp-image-get-layer-by-name image "Fundo")))
                    (drawable (car (gimp-image-active-drawable image)))
                    (nfilename (string-append only_filename ".jpeg"))
                )
                (gimp-message filename)
                (gimp-image-scale-full image value value INTERPOLATION-NONE)
                (gimp-image-convert-grayscale image)
                (gimp-xcf-save RUN-NONINTERACTIVE image drawable filename filename)
                (gimp-file-save RUN-NONINTERACTIVE image drawable nfilename nfilename)                    
                (gimp-image-delete image)
            )
            (set! filelist (cdr filelist))
        )
	)
)
(script-fu-register "rescale"
	""
	"Do nothing"
	"Joey User"
	"Joey User"
	"August 2000"
	""
)