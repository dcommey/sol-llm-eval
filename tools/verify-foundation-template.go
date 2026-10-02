// Verify the actual Ollama Go template against independently rendered publisher Jinja.
package main
import ("bytes"; "encoding/json"; "fmt"; "os"; "strings"; "text/template")
type Message struct { Role string `json:"role"`; Content string `json:"content"` }
func main() {
 source,err:=os.ReadFile(os.Args[1]);if err!=nil {panic(err)}
 pieces:=strings.Split(string(source),`TEMPLATE """`);if len(pieces)!=2 {panic("No unique template")}
 body:=strings.Split(pieces[1],`"""`)[0]
 tpl,err:=template.New("native").Parse(body);if err!=nil {panic(err)}
 var cases [][]Message
 if err=json.NewDecoder(os.Stdin).Decode(&cases);err!=nil {panic(err)}
 rendered:=[]string{}
 for _,messages:=range cases {var output bytes.Buffer;err=tpl.Execute(&output,map[string]any{"Messages":messages});if err!=nil {panic(err)};rendered=append(rendered,output.String())}
 data,err:=json.Marshal(rendered);if err!=nil {panic(err)};fmt.Println(string(data))
}
