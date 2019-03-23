const express = require('express')
const cookieParser = require('cookie-parser')
const bodyParser = require('body-parser');
const jwt = require('express-jwt');
const  multer  = require('multer')
const path = require('path')
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/')
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname)) //Appending extension
  }
})
var upload = multer({ storage: storage })

app = express();

app.use(cookieParser())
app.use(bodyParser.json());
app.use(express.static('public'))

app.post('/upload', upload.single('sketch'), function (req, res, next) {
  // req.file is the `avatar` file
  // req.body will hold the text fields, if there were any
  console.log(req.file.path)
  setTimeout(function() {
      res.send("Success");
}, 2000);

})


app.listen(3030, function() {
  console.log('Express Server listening on port 3030!');
})
